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
- prostate
```angular2html
         BN    WD     MD    PD     SUM
Train  2076  6303  4541  2383    15303
Valid   666   923   573   320     2482
Test    217  2463  4836   958     8474
```
- gastric
```angular2html
            BN    WD     MD    PD     SUM
Dataset                          
train     26010  9403  11202  13003   59618
valid     8780  7083    5177   7417   28457
test      11304  7927   6667   9060   34958
```



# Experiments & Tasks
## Tasks
- Single-Organ Model
```
python colon.py --epochs 30 --lr 0.001 --batch_size 40

RESULT
------------------------------
Accuracy : 0.868         
                         
Recall for Benign : 0.997
Precision for Benign : 0.999
f1-score for Benign : 0.499

Recall for WD : 0.176
Precision for WD : 0.219
f1-score for WD : 0.098

Recall for MD : 0.915
Precision for MD : 0.872
f1-score for MD : 0.446

Recall for PD : 0.789
Precision for PD : 0.901
f1-score for PD : 0.421

Precision for Cancer : 0.999
f1-score for Cancer : 0.500

------------------------------
Confusion Matrix :
label     0     1      2     3
infer
0      6703     0      4     0
1        14   349   1099   131
2         1  1637  14739   535
3         3     0    271  2491
```


```
python prostate.py --epochs 30 --lr 0.001 --batch_size 40

------------------------------
Accuracy : 0.533            
                            
Recall for Benign : 0.599   
Precision for Benign : 0.474
f1-score for Benign : 0.265 
                            
Recall for WD : 0.507       
Precision for WD : 0.626    
f1-score for WD : 0.280     

Recall for MD : 0.535       
Precision for MD : 0.713    
f1-score for MD : 0.306     

Recall for PD : 0.573
Precision for PD : 0.213
f1-score for PD : 0.155

Recall for Cancer : 0.983
Precision for Cancer : 0.989
f1-score for Cancer : 0.493

------------------------------
Confusion Matrix :
label    0     1     2    3
infer
0      130    65    62   17
1       37  1249   677   31
2        2   678  2589  361
3       48   471  1508  549

```
- low performance -> apply high aug
```angular2html
python prostate.py --epochs 30 --lr 0.001 --batch_size 40 -tf high

```
```
python gastric.py --epochs 30 --lr 0.001 --batch_size 40

------------------------------
Accuracy : 0.795

Recall for Benign : 0.948
Precision for Benign : 0.944
f1-score for Benign : 0.473

Recall for WD : 0.784
Precision for WD : 0.801
f1-score for WD : 0.396

Recall for MD : 0.487
Precision for MD : 0.579
f1-score for MD : 0.265

Recall for PD : 0.840
Precision for PD : 0.743
f1-score for PD : 0.394

Recall for Cancer : 0.973
Precision for Cancer : 0.975
f1-score for Cancer : 0.487

------------------------------
Confusion Matrix :
label    0.0   1.0   2.0   3.0
infer
0.0    10721   133   219   282
1.0      466  6213   959   122
2.0       97  1214  3250  1049
3.0       20   367  2239  7607

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
- [ ] Improve Prostate's performance 

