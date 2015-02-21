---
title: "PML Assignment - WLE Analysis"
author: "Yonatan G"
date: "Thursday, February 19, 2015"
output: html_document
---

Analysis of the Weight Lifting Exercises Data set

Executive Summary

Human Activity Recognition(HAR) data was collected and provided with the following address http://groupware.les.inf.puc-rio.br/har. The approach we propose for the Weight Lifting Exercises data set is to investigate "how (well)" an activity was performed by the wearer.

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3SCQQCHYx

In this assignment, we will build a predictive model to determine whether a particular form of exercise (barbell lifting) is performed correctly, using accelerometer data.

Getting and Cleaning the Data + Exploratory Data Analysis

Assuming the training and testing data sets are downloaded and saved to the working directory, both will be read into respective data frames, empty fields will be read as NA's:

```{r}
setwd("C:/Users/Yonatan/Coursera")
pmlTrain <- read.csv("pml-training.csv", header=TRUE, na.strings=c("","NA"))
pmlTest <- read.csv("pml-testing.csv", header=TRUE, na.strings=c("","NA"))
dim(pmlTrain)
dim(pmlTest)
```

There are 19622 observations and 160 variables for the training data and 20 observations and 160 variables for the test data. One of the variables, "classe" is the output which will be predicted with the test data.

Let us see the completeness of the data first:

```{r}
sum(complete.cases(pmlTrain))
```

Out of the huge training data frame only a limited number of observations are complete, hence we need to filter out the complete cases and consider those for the analysis, this will be done on both training and test data:
```{r}
pmlTrain <- pmlTrain[ , colSums(is.na(pmlTrain)) == 0]
pmlTest <- pmlTest[ , colSums(is.na(pmlTest)) == 0]
dim(pmlTrain)
```


Now we have 60 variables. Further exploration of the data revels that there are still some variables that can be reduced, for example the order number(X), name of the participant, time of measurement and "new-window" variable that is related to most of the empty fields. Reduction of variables will be done next based on variable titles:

```{r}
pmlTrain <- pmlTrain[,!grepl("X|user_name|timestamp|new_window", colnames(pmlTrain))]
pmlTest <- pmlTest[,!grepl("X|user_name|timestamp|new_window", colnames(pmlTest))]
dim(pmlTrain)
```

Now we are down to 54 variables.

We now split the updated training data set into a training data set (70% of the observations "train_pml") and a validation dataset (30% of the observations "valid_pml"). This validation data set will allow us to perform cross validation when developing our model.

```{r}
library(caret)
inTrain = createDataPartition(y = pmlTrain$classe, p = 0.7, list = FALSE)
train_pml = pmlTrain[inTrain, ]
valid_pml = pmlTrain[-inTrain, ]
```

Next we want to observe the correlation of the variables, highly correlated variables might be removed afterwards:

```{r}
M <- abs(cor(train_pml[,-54]))
diag(M) <- 0
which(M > .9, arr.ind=T)
```

We can see that there are lots of related variables even with the threshold of 0.9. Let us see the graphical depiction of the correlation:

```{r}
library(corrplot)
corrplot(M, order="FPC",tl.col = rgb(0, 0, 0))
```

It is clear that there are lot's of highly correlated variables, therefore it is better to resort to the Principal Component Analysis method.

PCA Analysis

We pre-process our data using a principal component analysis, leaving out the last column ('classe'). After pre-processing, we use the 'predict' function to apply the pre-processing to both the training and validation subsets of the original larger 'training' data set.

```{r}
preProc <- preProcess(train_pml[, -54], method = "pca", thresh = 0.99)
trainPml <- predict(preProc, train_pml[, -54])
validPml <- predict(preProc, valid_pml[, -54])
```

Next we use the randomforest method to train a model using the train_pml data, which the subset of  the original training data. We use the parameters in the train function so that we can have a faster process. And set the seed for reproducibility:

```{r}
set.seed(33833)
modelFit <- train(train_pml$classe ~ ., method = "rf", data = trainPml, trControl = trainControl(method = "cv",number = 4), importance = TRUE) 
```

Cross-Validation and OOS Error

Once again we use the prediction function with the fitted model and the data set split for cross validation, then we can see the confusion matrix to see if the model can be validated:

```{r}
crossValid <- predict(modelFit, validPml)
confusM <- confusionMatrix(valid_pml$classe, crossValid)
confusM$table
```

We can see that we have managed to create a model that can predict the output pretty fairly, with some residual values. Let us now calculate the Out-of-Sample error for this prediction model, first the estimated accuracy of the model:

```{r}
accur <- postResample(valid_pml$classe, crossValid)[[1]]
accur
```

Hence the Out-of-Sample(OOS) error will be:

```{r}
1 - accur
```

Finally we can now predict the output of the original testing data set provided. First we pre-process the original data set and run our fitted model against it:

```{r}
testPml <- predict(preProc, pmlTest[, -54])
prediction <- predict(modelFit, testPml)
```

And the predictions are (drum roll please .......) 

```{r}
prediction
```