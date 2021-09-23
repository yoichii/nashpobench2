# NAS-HPO-Bench-II
The first benchmark dataset for joint CNN and training HPs optimization.
This code is for training all the models in the search space.
The people who want to use the benchmark dataset should access [the repo](https://github.com/yoichii/nashpobench2api).

## Usage
This code works on [ABCI](https://abci.ai).

Run
```
 ./init_abci.sh
```
for initialization.

Then, run 
```
 qsub acml12.sh
```
for creating 12 epoch benchmark data, or
```
 qsub acml200.sh
```
for 200 epoch data.
