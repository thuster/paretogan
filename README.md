# Pareto GAN

## Install dependencies
```
pip install torch numpy matplotlib pandas scipy
```
Note: we recommend installing torch with GPU support

## Run an experiment
```
python exps.py -ds 3 -type pareto
python exps.py -ds 3 -type normal
```

## Options
GAN type (-type): 
 * pareto
 * uniform
 * normal
 * lognormal

Dataset (-ds): 
 * 0: Keystrokes
 * 1: Wiki Traffic
 * 2: Live Journal
 * 3: Dual Cauchy

Note: real datasets may not be available anymore. Dual Cauchy is a good "dataset" to illustrate the concept. 
