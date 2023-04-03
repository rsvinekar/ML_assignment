library(caret)
training <- read.csv("pml-training.csv")
outsample <- read.csv("pml-testing.csv")
contains_any_na = sapply(outsample, function(x) any(is.na(x)))
training <- training[,!contains_any_na]
training <- training[training$new_window=="no",]
training <- training[,-c(1,3,4,5,6)]
outsample <- outsample[,!contains_any_na]
outsample <- outsample[,-c(1,3,4,5,6)]
training$user_name <- as.factor(training$user_name)
training$classe <- as.factor(training$classe)
outsample$user_name <- as.factor(outsample$user_name)
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10)
set.seed(12345)
gbmFit1 <- train(classe ~ ., data = training, 
                 method = "gbm", 
                 trControl = fitControl,
                 verbose = TRUE)
nbFit1 <- train(classe ~ ., data = training, 
                 method = "nb", 
                 trControl = fitControl,
                 verbose = TRUE)
predict(gbmFit1, newdata = outsample, type = "raw")
predict(nbFit1, newdata = outsample, type = "raw")