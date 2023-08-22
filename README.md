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
- total
```angular2html
            BN    WD     MD        PD        SUM
Dataset                          
train     50255  20515   49650     21973    142393
valid     14492  9605    17675     9968      51740
test      18242  12376   27616     13175     71409

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

For Cancer
------------------------------
Cancer Accuracy : 0.832

Recall for Benign : 0.986
Precision for Benign : 0.939
f1-score for Benign : 0.481

Recall for WD : 0.662
Precision for WD : 0.767
f1-score for WD : 0.355

Recall for MD : 0.825
Precision for MD : 0.780
f1-score for MD : 0.401

Recall for PD : 0.779
Precision for PD : 0.823
f1-score for PD : 0.400

Recall for Cancer : 0.976
Precision for Cancer : 0.994
f1-score for Cancer : 0.493

------------------------------
Confusion Matrix :
label    0.0   1.0    2.0    3.0
infer
0.0    18383   431    322    436
1.0      224  7811   2018    127
2.0       29  3419  20413   2295
3.0       16   144   2004  10086
------------------------------

For Organ
------------------------------
Organ Accuracy : 1.000

Recall for colon : 1.000
Precision for colon : 1.000
f1-score for colon : 0.500

Recall for prostate : 1.000
Precision for prostate : 0.998
f1-score for prostate : 0.500

Recall for gastric : 1.000
Precision for gastric : 1.000
f1-score for gastric : 0.500

------------------------------
Confusion Matrix :
organ          0.0   1.0    2.0
infer_organ
0.0          27968     0      0
1.0              9  5222      0
2.0              0     1  34958


## Result for Colon
------------------------------
Accuracy : 0.851

Recall for Benign : 0.997
Precision for Benign : 0.998
f1-score for Benign : 0.499

Recall for WD : 0.207
Precision for WD : 0.291
f1-score for WD : 0.121

Recall for MD : 0.930
Precision for MD : 0.834
f1-score for MD : 0.440

Recall for PD : 0.545
Precision for PD : 0.916
f1-score for PD : 0.342

Recall for Cancer : 0.999
Precision for Cancer : 0.999
f1-score for Cancer : 0.500

------------------------------
Confusion Matrix : 
label   0.0   1.0    2.0   3.0
infer                         
0.0    6704     1     10     0
1.0       9   412    965    32
2.0       7  1572  14983  1404
3.0       1     1    155  1721

## Result for Prostate
------------------------------
Accuracy : 0.771

Recall for Benign : 0.974
Precision for Benign : 0.884
f1-score for Benign : 0.464

Recall for WD : 0.790
Precision for WD : 0.773
f1-score for WD : 0.391

Recall for MD : 0.679
Precision for MD : 0.735
f1-score for MD : 0.353

Recall for PD : 0.798
Precision for PD : 0.750
f1-score for PD : 0.387

Recall for Cancer : 0.983
Precision for Cancer : 0.996
f1-score for Cancer : 0.495

------------------------------
Confusion Matrix : 
label  0.0   1.0   2.0  3.0
infer                      
0.0    611    47    30    3
1.0      8  1495   418   12
2.0      8   344  1342  132
3.0      0     6   187  580

## Result for Gastric
------------------------------
Accuracy : 0.825

Recall for Benign : 0.979
Precision for Benign : 0.910
f1-score for Benign : 0.472

Recall for WD : 0.745
Precision for WD : 0.865
f1-score for WD : 0.400

Recall for MD : 0.613
Precision for MD : 0.642
f1-score for MD : 0.314

Recall for PD : 0.859
Precision for PD : 0.811
f1-score for PD : 0.417

Recall for Cancer : 0.954
Precision for Cancer : 0.990
f1-score for Cancer : 0.486

------------------------------
Confusion Matrix : 
label    0.0   1.0   2.0   3.0
infer                         
0.0    11068   383   282   433
1.0      207  5904   635    83
2.0       14  1503  4088   759
3.0       15   137  1662  7785
```
- DANN (epoch 9 only)
```
python dann.py --epochs 30 --lr 0.001 --batch_size 16

for Cancer
------------------------------
Accuracy : 0.216

Recall for Benign : 0.633
Precision for Benign : 0.266
f1-score for Benign : 0.187

Recall for WD : 0.000
Precision for WD : 0.000
f1-score for WD : 0.000

Recall for MD : 0.050
Precision for MD : 0.132
f1-score for MD : 0.036

Recall for PD : 0.128
Precision for PD : 0.115
f1-score for PD : 0.061

Recall for Cancer : 0.341
Precision for Cancer : 0.711
f1-score for Cancer : 0.230

------------------------------
Confusion Matrix : 
label    0.0   1.0    2.0   3.0
infer                          
0.0    11805  8017  18308  6317
1.0        0     1      3     0
2.0      855  2305   1235  4973
3.0     5992  1482   5211  1654
------------------------------

for Organ
------------------------------
Accuracy : 0.535

Recall for colon : 0.388
Precision for colon : 0.577
f1-score for colon : 0.232

Recall for prostate : 0.099
Precision for prostate : 0.029
f1-score for prostate : 0.022

Recall for gastric : 0.718
Precision for gastric : 0.800
f1-score for gastric : 0.378

------------------------------
Confusion Matrix : 
organ          0.0   1.0    2.0
infer_organ                    
0.0          10846   202   7745
1.0          15362   515   2129
2.0           1769  4506  25084


```
-> too low accuracy with epoch 9 -> cancel the project and added PCGrad

- DANN with PCGrad
```angular2html
python train_pcgrad.py --batch_size 16

For Cancer
------------------------------
Accuracy : 0.322

Recall for Benign : 0.874
Precision for Benign : 0.311
f1-score for Benign : 0.230

Recall for WD : 0.000
Precision for WD : 0.000
f1-score for WD : 0.000

Recall for MD : 0.192
Precision for MD : 0.371
f1-score for MD : 0.127

Recall for PD : 0.071
Precision for PD : 0.308
f1-score for PD : 0.058

Recall for Cancer : 0.272
Precision for Cancer : 0.851
f1-score for Cancer : 0.206

------------------------------
Confusion Matrix : 
label    0.0   1.0    2.0   3.0
infer                          
0.0    16293  8210  19428  8409
2.0     1641  2811   4762  3614
3.0      718   784    567   921
------------------------------

For organ
------------------------------
Accuracy : 0.253

Recall for colon : 0.352
Precision for colon : 0.477
f1-score for colon : 0.202

Recall for prostate : 0.637
Precision for prostate : 0.087
f1-score for prostate : 0.076

Recall for gastric : 0.117
Precision for gastric : 0.449
f1-score for gastric : 0.093

------------------------------
Confusion Matrix : 
organ          0.0   1.0    2.0
infer_organ                    
0.0           9834   842   9925
1.0          14163  3329  20936
2.0           3980  1052   4097

```
-> It not works as we thought, gradient reversal layer to 'Organ Classifier' may be harmful to 'Cancer classifier' too even we used PCGrad 

-> especially no infer for label 1(WD)

## Sampling Data

### Data info
```
colon   0      1      2      3     sum
train   6476   4554   7817   4755   23602
valid   4087   1599   5134   2231   13051
test    5244   1986   7495   3157   17882

gastric   0    1      2      3      sum
train   6755   4942   4900   5875   22472
valid   6494   5730   4623   6426   23273
test   11304   7927   6667   9060   34958

prostate   0   1      2     3      sum
train   2076   6303   451   2283   11113
valid   666    923    573   320    2482
test    217    2463   4836  958    8474
               
total   0       1       2       3       sum
train   15307   15799   13168   12913   57187
valid   11247   8252    10330   8977    38806
test    16765   12376   18998   13175   61314
```

### experiments

- total_organ
```
python total_organ.py --sample

 ------------------------------
Accuracy : 0.788

Recall for Benign : 0.974
Precision for Benign : 0.888
f1-score for Benign : 0.465

Recall for WD : 0.740
Precision for WD : 0.602
f1-score for WD : 0.332

Recall for MD : 0.681
Precision for MD : 0.815
f1-score for MD : 0.371

Recall for PD : 0.770
Precision for PD : 0.796
f1-score for PD : 0.391

Recall for Cancer : 0.954
Precision for Cancer : 0.990
f1-score for Cancer : 0.486
------------------------------
Confusion Matrix :
label    0.0   1.0    2.0   3.0
infer
0.0    18168   317    998   976
1.0      425  8738   4765   580
2.0       30  2365  16854  1423
3.0       29   385   2140  9965


## For Colon
------------------------------
Accuracy : 0.807

Recall for Benign : 0.996
Precision for Benign : 0.989
f1-score for Benign : 0.496

Recall for WD : 0.306
Precision for WD : 0.205
f1-score for WD : 0.123

Recall for MD : 0.827
Precision for MD : 0.843
f1-score for MD : 0.418

Recall for PD : 0.617
Precision for PD : 0.802
f1-score for PD : 0.349

Recall for Cancer : 0.997
Precision for Cancer : 0.999
f1-score for Cancer : 0.499

------------------------------
Confusion Matrix : 
label   0.0   1.0    2.0   3.0
infer                         
0.0    6693     5     63     6
1.0      15   608   2255    93
2.0      13  1358  13330  1110
3.0       0    15    465  1948

## For Prostate
------------------------------
Accuracy : 0.764

Recall for Benign : 0.954
Precision for Benign : 0.920
f1-score for Benign : 0.468

Recall for WD : 0.800
Precision for WD : 0.758
f1-score for WD : 0.389

Recall for MD : 0.666
Precision for MD : 0.726
f1-score for MD : 0.347

Recall for PD : 0.770
Precision for PD : 0.733
f1-score for PD : 0.376

Recall for Cancer : 0.989
Precision for Cancer : 0.994
f1-score for Cancer : 0.496

------------------------------
Confusion Matrix : 
label  0.0   1.0   2.0  3.0
infer                      
0.0    598    34    18    0
1.0     20  1514   453   10
2.0      9   330  1316  157
3.0      0    14   190  560

## For Gastric
------------------------------
Accuracy : 0.777

Recall for Benign : 0.962
Precision for Benign : 0.834
f1-score for Benign : 0.447

Recall for WD : 0.835
Precision for WD : 0.694
f1-score for WD : 0.379

Recall for MD : 0.331
Precision for MD : 0.724
f1-score for MD : 0.227

Recall for PD : 0.823
Precision for PD : 0.800
f1-score for PD : 0.406

Recall for Cancer : 0.908
Precision for Cancer : 0.981
f1-score for Cancer : 0.472

------------------------------
Confusion Matrix : 
label    0.0   1.0   2.0   3.0
infer                         
0.0    10877   278   917   970
1.0      390  6616  2057   477
2.0        8   677  2208   156
3.0       29   356  1485  7457

```

- multi_task
```
python multi_task.py --sample

# For Cancer
------------------------------
Accuracy : 0.773

Recall for Benign : 0.961
Precision for Benign : 0.863
f1-score for Benign : 0.455

Recall for WD : 0.714
Precision for WD : 0.579
f1-score for WD : 0.320

Recall for MD : 0.653
Precision for MD : 0.802
f1-score for MD : 0.360

Recall for PD : 0.786
Precision for PD : 0.802
f1-score for PD : 0.397

Recall for Cancer : 0.943
Precision for Cancer : 0.984
f1-score for Cancer : 0.482

------------------------------
Confusion Matrix :
label    0.0   1.0    2.0    3.0
infer
0.0    17916   365   1524    951
1.0      673  8425   4840    616
2.0       17  2767  16171   1208
3.0       46   248   2222  10169
------------------------------

# For Organ
------------------------------
Accuracy : 0.996

Recall for colon : 1.000
Precision for colon : 0.992
f1-score for colon : 0.498

Recall for prostate : 1.000
Precision for prostate : 0.992
f1-score for prostate : 0.498

Recall for gastric : 0.993
Precision for gastric : 1.000
f1-score for gastric : 0.498

------------------------------
Confusion Matrix :
organ          0.0   1.0    2.0
infer_organ
0.0          27970     0    214
1.0              7  5223     33
2.0              0     0  34711


## For Colon
------------------------------
Accuracy : 0.818

Recall for Benign : 0.998
Precision for Benign : 0.996
f1-score for Benign : 0.499

Recall for WD : 0.333
Precision for WD : 0.223
f1-score for WD : 0.133

Recall for MD : 0.831
Precision for MD : 0.854
f1-score for MD : 0.421

Recall for PD : 0.671
Precision for PD : 0.811
f1-score for PD : 0.367

Recall for Cancer : 0.999
Precision for Cancer : 0.999
f1-score for Cancer : 0.500

------------------------------
Confusion Matrix : 
label   0.0   1.0    2.0   3.0
infer                         
0.0    6708     3     15     6
1.0       9   661   2239    57
2.0       3  1302  13387   976
3.0       1    20    472  2118

## For Prostate
------------------------------
Accuracy : 0.723

Recall for Benign : 0.946
Precision for Benign : 0.886
f1-score for Benign : 0.458

Recall for WD : 0.726
Precision for WD : 0.732
f1-score for WD : 0.364

Recall for MD : 0.612
Precision for MD : 0.698
f1-score for MD : 0.326

Recall for PD : 0.824
Precision for PD : 0.636
f1-score for PD : 0.359

Recall for Cancer : 0.983
Precision for Cancer : 0.993
f1-score for Cancer : 0.494

------------------------------
Confusion Matrix : 
label  0.0   1.0   2.0  3.0
infer                      
0.0    593    55    18    3
1.0     29  1374   454   21
2.0      4   416  1210  104
3.0      1    47   295  599

## For Gastric
------------------------------
Accuracy : 0.745

Recall for Benign : 0.939
Precision for Benign : 0.795
f1-score for Benign : 0.430

Recall for WD : 0.806
Precision for WD : 0.658
f1-score for WD : 0.362

Recall for MD : 0.236
Precision for MD : 0.570
f1-score for MD : 0.167

Recall for PD : 0.823
Precision for PD : 0.816
f1-score for PD : 0.410

Recall for Cancer : 0.884
Precision for Cancer : 0.968
f1-score for Cancer : 0.462

------------------------------
Confusion Matrix : 
label    0.0   1.0   2.0   3.0
infer                         
0.0    10615   307  1491   942
1.0      635  6390  2147   538
2.0       10  1049  1574   128
3.0       44   181  1455  7452

```
- DANN
```
python dann.py --sample --batch_size 16

```
- DANN with PCGrad
```
python train_pcgrad.py --sample --batch_size 16

```



# todo
- [x]  dataloader for each organs
- [x] classification model for each organ
- [x] dataloader for integrated model
- [x] classification for integrated model
- [x] experiments for single and integrated models
- [x] think of how using data
- [x] apply DANN
- [x] experiment with undersampled data
- [ ] Visulaize DANN with T-SNE
- [ ] apply MixStyle
