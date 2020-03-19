# LP-SSL-text-classification
NYU DS-GA 1012 final project; idea adopted from CVPR 2019 paper Label Propagation for Deep Semi-supervised Learning by Iscen et al (https://ieeexplore.ieee.org/abstract/document/8954421).

## train baseline
```
python train_baseline.py -t baseline -m 20
```

## train fully supervised
```
python train_fully_supervised.py -t supervised -m 20
```

## train phase2
```
python train_phase2.py -t phase2 -m 50
```