# General-Cancer-Classification
```
TeamBasic Project - General Cancer classification
```

### Team Basic
[Dong Yeon Kang](https://github.com/Dong-Yeon-Kang), [Sang Hyeok Shin](https://github.com/SSH0515), [Jong Woo Lim](https://github.com/imngooh) Korea University, EE
 
# Introduction
- general cancer classification Using deep learning, especially CNNs
- previous researches (pan-cancer classification) usually based on gene expressions and its characteristics
- or CNNs used for only single organ's cancer classification (Ex. breast, colon, prostate, gastric ...)
- We aim to make 'General' model to cancer classification
- Get cell images, for any organs and classify its cancer grade.

# Progress
1. Baseline Models
   - make single-organ cancer classification model for colon, prostate, gastric cancer classification
2. Integrated Models
   - Just use one model for all datas, learn and infer. Compare to single organ models its performance. 
3. Learning strategy(domain adaptation, generalization)
   - Use ensembles, DANN, MixStyle, etc..

# Dataset Info
- colon
```angular2html
            BN    WD     MD    PD     SUM
Dataset                          
train    16129  4809  33947  6507   61392
valid     5046  1599  11925  2231   20801
test      6721  1986  16113  3157   27977
```

- gastric
```angular2html
            BN    WD     MD    PD     SUM
Dataset                          
train    27707  10196 14999  12046   64948
valid     8398  2239  2370   2374    15381
test      7955  1795  2458   3579    15787
```
<<<<<<< HEAD
# Experiments & Tasks
## Tasks
- Single-Organ Model
```
python colon.py --epochs 30 --lr 0.001 --batch_size 40
```
```
python prostate.py --epochs 30 --lr 0.001 --batch_size 40
```
```
python gastric.py --epochs 30 --lr 0.001 --batch_size 40
```
- Total-organ Model
```
python total_organ.py --epochs 30 --lr 0.001 --batch_size 40
```
- Multi-task Model
```
python multi_task.py --epochs 30 --lr 0.001 --batch_size 40
```
- DANN
```
python dann.py --epochs 30 --lr 0.001 --batch_size 40
```


# todo
- [x]  dataloader for each organs -> code finished, not experimented
- [x] classification model for each organ -> code finished, not experimented
- [x] dataloader for integrated model -> code finished, not experimented
- [x] classification for integrated model -> code finished, not experimented
- [ ] experiments for single and integrated models
- [ ] think of how using data
- [x] apply DANN -> code finished, not experimented
- [ ] Visulaize with T-SNE
- [ ] apply MixStyle

