ML Assignment
================
Rithvik
2023-04-04

# Introduction

This document describes use of ML models for prediction of exercise type
in a given dataset called the Weight Lifting Exercise Dataset. The data
was collected from accelerometers on the belt, forearm, arm, and dumbell
of 6 participants. They were asked to perform barbell lifts correctly
and incorrectly in 5 different ways.
<http://groupware.les.inf.puc-rio.br/har> For this purpose, we need to
load the dataset, train an ML model and use the model to predict the
exercise method for the test data.

# Data loading and cleaning

## Download data

The data is downloaded and loaded as `maindata` and `outsample` using
read.csv and default options.

``` r
maindata <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
outsample <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
```

## Clean data

We can use `View()` in RStudio to check the data. `dim(maindata)=`19622,
160 and `dim(outsample)=`20, 160. The number of columns is the same,
however maindata has a `classe` column, while outsample has a dummy
variable `problem_id` which is a counter 1:20. We note that many columns
are having NA values, except when the `new_window` column has value
`yes`. In the latter case, there are values instead of NA. The outsample
data has no `new_window` values given as `yes`. Thus it is better to
filter out all entries where `new_window=="yes"`. The number of entries
with `new_window=="yes"` is very small,
i.e.`dim(maindata[maindata$new_window=="yes",])=`406, 160 compared to
19216 entries for `new_window=="no"`. Since the function of the
`new_window` column is not clear, we will use the `new_window=="no"`
entries only.

``` r
maindata <- maindata[maindata$new_window=="no",]
```

This makes several columns with NA values. These can be removed
immediately from both maindata and outsample. We will check that no
extra columns are removed and the maindata and sample match.

``` r
contains_any_na = sapply(outsample, function(x) any(is.na(x)))
maindata <- maindata[,!contains_any_na]
outsample <- outsample[,!contains_any_na]
```

The first 6 columns are removed and preserved in another variable in
case we need it later. The same procedure is used on both the maindata
and outsample.

``` r
maindata_1st6 <- maindata[,c(1:6)] 
maindata <- maindata[,-c(1:6)]
```

``` r
outsample_1st6 <- outsample[,c(1:6)]
outsample <- outsample[,-c(1:6)]
```

We can make factor variable out of the classe column.

``` r
maindata$classe <- as.factor(maindata$classe)
```

The new dataset is now defined as dim(maindata)=19216, 54. And the
outsample is dim(outsample)=20, 54. Thus we have 54 variables in both.
There are `dim(maindata)[1]` entries in the maindata for training.

# Machine learning

We first load the library needed

``` r
library(caret)
```

First we will split the maindata into a training and test set, so we can
choose the best machine learning strategy.

``` r
inTrain <- createDataPartition(maindata$classe, p = 3/4)[[1]]
training <- maindata[inTrain,]
testing <- maindata[-inTrain,]
```

Using this strategy we can try many methods

``` r
rpartFit <- train(classe ~ ., data = training, method = "rpart")
```

``` r
pred_rpart <- predict(rpartFit, newdata = testing[,-55], type = "raw")
```

``` r
cMpred_rpart <- confusionMatrix(data=pred_rpart, reference=testing$classe)
cMpred_rpart$table
```

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1225  378  370  339  105
    ##          B   33  319   25  146  133
    ##          C  108  232  443  301  225
    ##          D    0    0    0    0    0
    ##          E    1    0    0    0  419

``` r
cMpred_rpart$overall
```

    ##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
    ##   5.010412e-01   3.491708e-01   4.867983e-01   5.152829e-01   2.846731e-01 
    ## AccuracyPValue  McnemarPValue 
    ##  2.354844e-218            NaN

Clearly the Classification and Regression Trees (CART) model is not very
accurate with 50.1041233% accuracy by default. It may work with some
tweaking. Other methods like “glm” or “lm” do not work as the number of
variables is very large, or in case of “glm”, the output levels are too
many.

Boost methods combine many different small regressors using different
strategies. One commonly used boost method which requires very little
tweaking from default is the “gbm” or Gradient boosting machine. We can
use this technique as it is perfect for the use-case we have.

``` r
gbmFit <- train(classe ~ ., data = training, 
                 method = "gbm",
                 verbose = FALSE)
```

``` r
pred_gbm <- predict(gbmFit, newdata = testing[,-55], type = "raw")
```

``` r
cMpred_gbm <- confusionMatrix(data=pred_gbm, reference=testing$classe)
cMpred_gbm$table
```

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1365    3    0    0    0
    ##          B    1  916    6    5    2
    ##          C    0    7  829    7    2
    ##          D    1    3    1  774    5
    ##          E    0    0    2    0  873

``` r
cMpred_gbm$overall[1:2]
```

    ##  Accuracy     Kappa 
    ## 0.9906289 0.9881459

We can see that the training confusionmatrix gives 99.0628905% accuracy.
This is much better than “rpart”. Thus we can use the “gbm” model for
our final analysis

# Cross Validation

There are many methods for cross validation. One we have already used,
by partitioning the data into training and testing sets. However, we can
use more rigorous methods on the whole maindata so that we can have a
model which we can then use on the outsample. We can use “cv” or
“repeatcv” or k-fold cross validations. We can use K=10 and 10 repeats
of the same with “repeatcv”, and then take the average to remove any
bias in sampling.

``` r
fitControl <- trainControl( method = "repeatedcv",number = 10, repeats = 10)
set.seed(1729)
```

The seed can be anything, including the [taxicab
number](https://en.wikipedia.org/wiki/Taxicab_number).

``` r
gbmFit1 <- train(classe ~ ., data = maindata, 
                 method = "gbm", 
                 trControl = fitControl,
                 verbose = FALSE)
```

Now we can test the model. Since the outsample does not give expected
outcome, we can use the testing set we had before. **HOWEVER**, this is
**inaccurate**, as the testing set is part of the maindata set which is
the training set for the model. Thus we can expect the numbers to be
better than the previous case. We can check it out anyway:

``` r
pred_gbm1 <- predict(gbmFit1, newdata = testing[, -55], type = "raw")
```

``` r
cMpred_gbm1 <- confusionMatrix(data=pred_gbm1, reference=testing$classe)
cMpred_gbm1$table
```

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1367    6    0    0    0
    ##          B    0  917    2    4    2
    ##          C    0    6  833    4    1
    ##          D    0    0    2  778    2
    ##          E    0    0    1    0  877

``` r
cMpred_gbm1$overall[1:2]
```

    ##  Accuracy     Kappa 
    ## 0.9937526 0.9920962

The 99.3752603% accuracy is much better, but this is expected as the
training set contained the test set, so this increase is really
meaningless. The model itself can be checked by its plots

``` r
ggplot(gbmFit1)
```

![We can see the increase in estimated accuracy as the boosting
iterations are increased](ml_files/figure-gfm/unnamed-chunk-16-1.png)

The plots indicate a very high level of accuracy $>99$%, and the out of
sample error should thus be very low. Since we do not know the expected
outcome, it is difficult to verify, but it is likely the error is
negligible, so long as the variables are measured in the same
conditions.

# Application of the model

The model can now be applied to our ‘outsample’

``` r
pred_prob <- predict(gbmFit1, newdata = outsample, type = "prob")
pred_prob
```

    ##              A           B           C           D           E
    ## 1  0.033040721 0.741904611 0.074388315 0.124448729 0.026217625
    ## 2  0.963376407 0.022463159 0.006724335 0.002474260 0.004961839
    ## 3  0.086674482 0.764112643 0.073339417 0.025902206 0.049971252
    ## 4  0.977201649 0.003466924 0.007744126 0.010331798 0.001255503
    ## 5  0.964052169 0.020517439 0.007199514 0.003203078 0.005027800
    ## 6  0.003243891 0.034362431 0.043298497 0.023433657 0.895661524
    ## 7  0.014690747 0.055825630 0.055952131 0.835917978 0.037613515
    ## 8  0.030809868 0.609817504 0.045581013 0.273177569 0.040614045
    ## 9  0.990889738 0.004900352 0.001416386 0.001557325 0.001236200
    ## 10 0.952591417 0.026443982 0.010538029 0.006392331 0.004034242
    ## 11 0.017669826 0.940786438 0.018205648 0.010656967 0.012681120
    ## 12 0.018603451 0.027412596 0.905714049 0.020497202 0.027772702
    ## 13 0.019819450 0.898983430 0.014497135 0.020633922 0.046066063
    ## 14 0.992136850 0.002679342 0.002681833 0.001039602 0.001462373
    ## 15 0.003829300 0.019783358 0.008456394 0.032019884 0.935911064
    ## 16 0.007426780 0.010303652 0.002046778 0.011036932 0.969185859
    ## 17 0.934320197 0.006938441 0.007625800 0.002439096 0.048676466
    ## 18 0.050158879 0.864021036 0.005233146 0.058501812 0.022085127
    ## 19 0.234776584 0.682581228 0.013859519 0.054279153 0.014503515
    ## 20 0.003519166 0.985268288 0.002203913 0.002613723 0.006394910

We can see some columns having $p>0.6$, which are the predictions of the
model.

``` r
pred_mat <- pred_prob>0.6
pred_mat
```

    ##           A     B     C     D     E
    ##  [1,] FALSE  TRUE FALSE FALSE FALSE
    ##  [2,]  TRUE FALSE FALSE FALSE FALSE
    ##  [3,] FALSE  TRUE FALSE FALSE FALSE
    ##  [4,]  TRUE FALSE FALSE FALSE FALSE
    ##  [5,]  TRUE FALSE FALSE FALSE FALSE
    ##  [6,] FALSE FALSE FALSE FALSE  TRUE
    ##  [7,] FALSE FALSE FALSE  TRUE FALSE
    ##  [8,] FALSE  TRUE FALSE FALSE FALSE
    ##  [9,]  TRUE FALSE FALSE FALSE FALSE
    ## [10,]  TRUE FALSE FALSE FALSE FALSE
    ## [11,] FALSE  TRUE FALSE FALSE FALSE
    ## [12,] FALSE FALSE  TRUE FALSE FALSE
    ## [13,] FALSE  TRUE FALSE FALSE FALSE
    ## [14,]  TRUE FALSE FALSE FALSE FALSE
    ## [15,] FALSE FALSE FALSE FALSE  TRUE
    ## [16,] FALSE FALSE FALSE FALSE  TRUE
    ## [17,]  TRUE FALSE FALSE FALSE FALSE
    ## [18,] FALSE  TRUE FALSE FALSE FALSE
    ## [19,] FALSE  TRUE FALSE FALSE FALSE
    ## [20,] FALSE  TRUE FALSE FALSE FALSE

And the predictions themselves can be easily obtained using
`type = "raw"`

# Conclusion

The final prediction is given in the **classe** column below:

``` r
pred_final <- predict(gbmFit1, newdata = outsample, type = "raw")
cbind(outsample_1st6[,2:3],classe=pred_final)
```

    ##    user_name raw_timestamp_part_1 classe
    ## 1      pedro           1323095002      B
    ## 2     jeremy           1322673067      A
    ## 3     jeremy           1322673075      B
    ## 4     adelmo           1322832789      A
    ## 5     eurico           1322489635      A
    ## 6     jeremy           1322673149      E
    ## 7     jeremy           1322673128      D
    ## 8     jeremy           1322673076      B
    ## 9   carlitos           1323084240      A
    ## 10   charles           1322837822      A
    ## 11  carlitos           1323084277      B
    ## 12    jeremy           1322673101      C
    ## 13    eurico           1322489661      B
    ## 14    jeremy           1322673043      A
    ## 15    jeremy           1322673156      E
    ## 16    eurico           1322489713      E
    ## 17     pedro           1323094971      A
    ## 18  carlitos           1323084285      B
    ## 19     pedro           1323094999      B
    ## 20    eurico           1322489658      B

Thanks!!
