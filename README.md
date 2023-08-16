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
- low performance -> apply high aug and small lr
```angular2html
python prostate.py --epochs 30 --lr 0.001 --batch_size 40 -tf high

------------------------------
Accuracy : 0.599

Recall for Benign : 0.616
Precision for Benign : 0.965
f1-score for Benign : 0.376

Recall for WD : 0.712
Precision for WD : 0.657
f1-score for WD : 0.342

Recall for MD : 0.417
Precision for MD : 0.585
f1-score for MD : 0.244

Recall for PD : 0.787
Precision for PD : 0.420
f1-score for PD : 0.274

Recall for Cancer : 0.997
Precision for Cancer : 0.950
f1-score for Cancer : 0.486

------------------------------
Confusion Matrix :
label    0     1    2    3
infer
0      386    10    4    0
1      175  1347  479   49
2       36   443  825  106
3       30    92  669  572

```

```
python prostate.py --epochs 30 --lr 0.0005 --batch_size 40 -tf high
Accuracy : 0.610

Recall for Benign : 0.689
Precision for Benign : 0.971
f1-score for Benign : 0.403

Recall for WD : 0.635
Precision for WD : 0.668
f1-score for WD : 0.326

Recall for MD : 0.506
Precision for MD : 0.563
f1-score for MD : 0.267

Recall for PD : 0.757
Precision for PD : 0.457
f1-score for PD : 0.285

Recall for Cancer : 0.997
Precision for Cancer : 0.959
f1-score for Cancer : 0.489

------------------------------
Confusion Matrix :
label    0     1     2    3
infer
0      432    11     2    0
1      136  1202   378   83
2       46   635  1000   94
3       13    44   597  550
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

------------------------------
Accuracy : 0.825

Recall for Benign : 0.974
Precision for Benign : 0.958
f1-score for Benign : 0.483

Recall for WD : 0.557
Precision for WD : 0.745
f1-score for WD : 0.319

Recall for MD : 0.862
Precision for MD : 0.753
f1-score for MD : 0.402

Recall for PD : 0.782
Precision for PD : 0.842
f1-score for PD : 0.405

Recall for Cancer : 0.984
Precision for Cancer : 0.990
f1-score for Cancer : 0.494

------------------------------
Confusion Matrix :
label    0.0   1.0    2.0    3.0
infer
0.0    18172   318    248    234
1.0      408  6580   1471    377
2.0       60  4723  21336   2210
3.0       12   184   1702  10123


# more anaylsis

## for organ(0) : colon
------------------------------
Accuracy : 0.855

Recall for Benign : 0.999
Precision for Benign : 0.996
f1-score for Benign : 0.499

Recall for WD : 0.042
Precision for WD : 0.167
f1-score for WD : 0.034

Recall for MD : 0.969
Precision for MD : 0.817
f1-score for MD : 0.443

Recall for PD : 0.475
Precision for PD : 0.931
f1-score for PD : 0.315

Recall for Cancer : 0.999
Precision for Cancer : 1.000
f1-score for Cancer : 0.500

------------------------------
Confusion Matrix : 
label   0.0   1.0    2.0   3.0
infer                         
0.0    6713     0     21     3
1.0       2    84    362    55
2.0       6  1900  15620  1599
3.0       0     2    110  1500

## for organ(1) : prostate
------------------------------
Accuracy : 0.756

Recall for Benign : 0.928
Precision for Benign : 0.922
f1-score for Benign : 0.463

Recall for WD : 0.668
Precision for WD : 0.794
f1-score for WD : 0.363

Recall for MD : 0.789
Precision for MD : 0.666
f1-score for MD : 0.361

Recall for PD : 0.746
Precision for PD : 0.821
f1-score for PD : 0.391

Recall for Cancer : 0.989
Precision for Cancer : 0.990
f1-score for Cancer : 0.495

------------------------------
Confusion Matrix : 
label  0.0   1.0   2.0  3.0
infer                      
0.0    582    31    12    6
1.0     26  1264   291   10
2.0     19   593  1560  169
3.0      0     4   114  542

## for organ(2) : gastric
------------------------------
Accuracy : 0.811

Recall for Benign : 0.962
Precision for Benign : 0.937
f1-score for Benign : 0.475

Recall for WD : 0.660
Precision for WD : 0.776
f1-score for WD : 0.357

Recall for MD : 0.623
Precision for MD : 0.606
f1-score for MD : 0.307

Recall for PD : 0.892
Precision for PD : 0.829
f1-score for PD : 0.430

Recall for Cancer : 0.969
Precision for Cancer : 0.982
f1-score for Cancer : 0.488

------------------------------
Confusion Matrix : 
label    0.0   1.0   2.0   3.0
infer                         
0.0    10877   287   215   225
1.0      380  5232   818   312
2.0       35  2230  4156   442
3.0       12   178  1478  8081


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

