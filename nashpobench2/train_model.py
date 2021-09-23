"""Train the model"""
from datetime import datetime
import logging
import time

from hydra.experimental import initialize, compose
import mlflow
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from nashpobench2.models import NASHPOModel, get_modules
from nashpobench2.utils.config import subcfg2instance
from nashpobench2.utils.dataloader import get_dataloader
from nashpobench2.utils.log import get_model_infos, get_weight


def train_model(
    cfg: DictConfig,
    cfgidx: dict,
    imgidx: dict
) -> float:
    """Train the model specified by the ``cfgidx'' variable.

    Args:
        cfg (DictConfig): possible training HP configurations.
        cfgidx (dict): a specific configuration (+ model arch).
        imgidx (dict): specify how to split image data.

    Examples:
        >>> with initialize(config_path='../config/hydra'):
        ...     cfg = compose(config_name='config')
        ...     with open('config/idx/split.yaml') as f:
        ...         imgidx = yaml.safe_load(f)
        >>> train_model(
        ...     cfg,
        ...     {'cellcode': '0|31|213', 'dataloaders': 0, 'datasets': 0, 'transforms': 0, 'optimizers': 0, 'lr': 3, 'momentum': 1, 'batch_size': 1, 'schedulers': 0, 'criteria': 0, 'num_blocks': 0, 'num_nodes': 0, 'num_cells': 0, 'channels': 0},
        ...     imgidx
        ... )
    """
    logger = logging.getLogger(__name__)
    # settings for reproductivity
    torch.manual_seed(cfg['training']['seed'])
    np.random.seed(cfg['training']['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True   # make (only) conv deterministic
    # get instances
    train_loader, valid_loader, test_loader = get_dataloader(
        cfg,
        imgidx,
        cfgidx
    )
    model = NASHPOModel(cfg, cfgidx)
    optimizer = get_modules(
        cfg,
        'optimizers',
        cfgidx,
        [optim],
        params=model.parameters()
    )
    scheduler = get_modules(
        cfg,
        'schedulers',
        cfgidx,
        [lr_scheduler],
        optimizer=optimizer
    )
    criterion = get_modules(
        cfg,
        'criteria',
        cfgidx,
        [nn]
    )
    # set cuda
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{cfg["training"]["gpu"]}')
        model.to(device)
        criterion.to(device)
        logger.debug(device)
        logger.debug(torch.backends.cudnn.enabled)
        torch.backends.cudnn.enabled = True
    # train
    test_acc = _train(model, optimizer, scheduler, criterion, train_loader, valid_loader, test_loader, cfg, cfgidx)

    return test_acc


def _train(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    criterion: _Loss,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    cfg: DictConfig,
    cfgidx: dict
) -> float:
    # weight
    #cos = nn.CosineSimilarity(dim=0)
    #init_weight = get_weight(model)
    #prev_cur_weights = [None,None,init_weight]

    # logger settings
    logger = logging.getLogger(__name__)
    logger.info(
            f'\n- cell: {cfgidx["cellcode"]}\n- optim: {optimizer}\n- scheduler: {scheduler.__class__.__name__}\n- criterion {criterion.__class__.__name__}\n- batch size {list(cfg["datasets"]["train"]["options"]["batch_size"])[cfgidx["batch_size"]]}\n- seed {cfg["training"]["seed"]}'
    )
    logger.info('====== train start ======')

    with mlflow.start_run(
        run_name=f'{list(cfg["datasets"]["train"]["options"]["batch_size"])[cfgidx["batch_size"]]}-{scheduler.get_last_lr()[0]}-{cfgidx["cellcode"]}'
    ):

        # log params
        flops, params = get_model_infos(model, (1, 3, 32, 32))
        mlflow.log_params(
            {
                'cellcode': cfgidx["cellcode"],
                'optimizer': optimizer.__class__.__name__,
                'scheduler': scheduler.__class__.__name__,
                'seed': cfg['training']['seed'],
                'params': params,
                'FLOPs': flops
            }
        )
        mlflow.log_params( {'batch_size': list(cfg['datasets']['train']['options']['batch_size'])[cfgidx['batch_size']]  })
        # optim settings
        mlflow.log_params({k:v for k,v in optimizer.param_groups[0].items() if k != 'params'})
        # scheduler settings
        badwords = ['optimizer', 'verbose', 'base_lrs', 'last_epoch']
        mlflow.log_params( {k:v for k,v in scheduler.__dict__.items() if not (k in badwords or k.startswith('_'))} )
        elpsdtime = []
        start = time.time()
        # train
        for epoch in range(cfg['training']['epochs']):
            model.train()
            running_loss = 0.0
            train_loss = 0.0
            logger.info(f'=== {epoch+1} epoch - lr: {scheduler.get_last_lr()} ===')
            elpsdtime.append(time.time())
            for step, (inputs, labels) in enumerate(train_loader):
                # cuda
                if torch.cuda.is_available():
                    device = torch.device(f'cuda:{cfg["training"]["gpu"]}')
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                # zero grad
                optimizer.zero_grad()
                # propagation
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            train_loss = running_loss / len(train_loader)
            elpsdtime.append(time.time())
            # train loss&acc
            train_loss_fixed, train_acc = _test(model, optimizer, scheduler, criterion, train_loader, cfg, 'train')
            logger.info(
                f'[{epoch+1}epoch - {elpsdtime[6*epoch+1]-elpsdtime[6*epoch]:.3f} sec -  train] acc: {train_acc:.3f}%, loss: {train_loss:.3f}({train_loss_fixed:.3f})'
            )
            # valid loss&acc
            elpsdtime.append(time.time())
            valid_loss, valid_acc = _test(model, optimizer, scheduler, criterion,  valid_loader, cfg, 'valid')
            elpsdtime.append(time.time())
            logger.info(
                f'[{epoch+1}epoch - {elpsdtime[6*epoch+3]-elpsdtime[6*epoch+2]:.3f} sec - valid] acc: {valid_acc:.3f}%, loss: {valid_loss:.3f}'
            )
            # test loss&acc
            elpsdtime.append(time.time())
            test_loss, test_acc = _test(model, optimizer, scheduler, criterion, test_loader, cfg, 'test')
            elpsdtime.append(time.time())
            logger.info(
                f'[{epoch+1}epoch - {elpsdtime[6*epoch+5]-elpsdtime[6*epoch+4]:.3f} sec - test] acc: {test_acc:.3f}%, loss: {test_loss:.3f}'
            )
            # log
            #prev_cur_weights[epoch%3] = get_weight(model)
            mlflow.log_metrics(
                {
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'test_loss': test_loss,
                    'train_acc': train_acc,
                    'valid_acc': valid_acc,
                    'test_acc': test_acc,
                    'train_time': round(elpsdtime[6*epoch+1]-elpsdtime[6*epoch], 3),
                    'valid_time': round(elpsdtime[6*epoch+3]-elpsdtime[6*epoch+2], 3),
                    'test_time': round(elpsdtime[6*epoch+5]-elpsdtime[6*epoch+4], 3),
                    'weight_cos':float(
                        cos(
                            prev_cur_weights[epoch%3] - prev_cur_weights[(epoch-1)%3],
                            prev_cur_weights[(epoch-1)%3] - prev_cur_weights[(epoch-2)%3]
                        )
                    ) if epoch != 0 else 0.0,
                    'weight_norm': float(
                        torch.norm(prev_cur_weights[epoch%3] - prev_cur_weights[(epoch-1)%3])
                    ),
                    'weight_distance': float(
                        torch.norm(prev_cur_weights[epoch%3] - init_weight)
                    ),
                },
                step=epoch
            )
        end = time.time()
        total_time = round(end-start, 3)
        mlflow.pytorch.log_model(model, 'model')
        mlflow.log_metric('total_time', total_time)

    return test_acc


def _test(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    criterion: _Loss,
    dataloader: DataLoader,
    cfg: DictConfig,
    mode: str
) -> float:
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            # cuda
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{cfg["training"]["gpu"]}')
                inputs = inputs.to(device)
                labels = labels.to(device)
            # propagation
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
    # print loss
    return running_loss/len(dataloader), correct/total*100


if __name__ == '__main__':
    import doctest
    doctest.testmod()
