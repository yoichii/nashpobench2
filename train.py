from datetime import datetime
import yaml

import hydra
from omegaconf import DictConfig
import mlflow

from nashpobench2.train_model import train_model
from nashpobench2.utils.config import get_nth_idx


@hydra.main(config_path='config/hydra')
def main(cfg: DictConfig):
    # set a path for logging (mlflow)
    epoch = cfg['training']['epochs']
    mlflow.set_tracking_uri(
        hydra.utils.to_absolute_path(f'log/acml{epoch}/mlflow')
    )
    # get the train-valid-test split idx of cifar10
    with open(hydra.utils.to_absolute_path('config/idx/split.yaml')) as f:
        imgidx = yaml.safe_load(f)
    # get the code of the model we train
    with open(
            hydra.utils.to_absolute_path(f'config/cellcodes/acml{epoch}.yaml')
            ) as f:
        cellcodes = yaml.safe_load(f)
    cellidx = cfg['architecture']['cellidx']
    cellcode = cellcodes[cellidx]
    # set experiment name
    mlflow.set_experiment(cellcode)
    # set training options (referred to as cfgidx)
    idx = cfg['training']['idx']
    cfgidx = get_nth_codes(cfg, idx)
    cfgidx.update({'cellcode': cellcode})
    # train
    train_model(
        cfg,
        cfgidx,
        imgidx
    )


if __name__ == '__main__':
    main()
