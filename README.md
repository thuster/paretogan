# Pareto GAN

## Install dependencies
```
pip install torch numpy matplotlib pandas scipy
```
Note: we recommend installing torch with GPU support

## Run an experiment
```
python exp -ds 3 -type pareto
python exp -ds 3 -type normal
```

## Options
GAN type: 
 * pareto
 * uniform
 * normal
 * lognormal

ds: 
 * 0: Keystrokes
 * 1: Wiki Traffic
 * 2: Live Journal
 * 3: Dual Cauchy

