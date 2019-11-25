#Loading the dataset
data_set<-read.csv("C:/Users/Somenath Banerjee/Desktop/projects/Loan_pred/train.csv")
#Create the data partition
index<-createDataPartition(data_set$Loan_Status,p=0.8,list = FALSE)
train_set<-data_set[index,]
test_set<-data_set[-index,]
#Basic summary statistics and looking for missing values
dim(train_set)
dim(test_set)
names(train_set)
library(dplyr)
glimpse(train_set)
sum(is.na(train_set))
sapply(train_set, function(x)sum(is.na(x)))

#Loan_status will be the target variable and logically Loan_ID will be insignificant
levels(train_set$Loan_Status)
train_set<-train_set[,-1]
test_set<-test_set[,-1]

#look at first few rows
head(train_set)#head(test_set)

#statistics and visualisation 
summary(train_set)
par(mfrow=c(1,2))
barplot(table(train_set$Loan_Status))
#Omit NA values
train_set<-na.omit(train_set)
test_set<-na.omit(test_set)

#credit history int to factor transformation
train_set$Credit_History<-as.factor(train_set$Credit_History)
test_set$Credit_History<-as.factor(test_set$Credit_History)

#set a control and train the models
library(caret)
library(randomForest)
control<-trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(99)
fit.glm<-train(Loan_Status~., train_set, method = "glm", metric = "Accuracy", trControl=control)
fit.glm
varImp(fit.glm)
set.seed(99)
fit.knn<-train(Loan_Status~., train_set, method = "knn", metric = "Accuracy", trControl=control)
set.seed(99)
fit.nb<-train(Loan_Status~., train_set, method = "nb", metric = "Accuracy", trControl=control)
set.seed(99)
fit.rpart<-train(Loan_Status~., train_set, method = "rpart", metric = "Accuracy", trControl=control)
set.seed(99)
fit.svm<-train(Loan_Status~., train_set, method = "svmRadial", metric = "Accuracy", trControl=control)
set.seed(99)
fit.rf<-train(Loan_Status~., train_set, method = "rf", metric = "Accuracy", trControl=control)
#or
fit.rf2<-randomForest(Loan_Status~., train_set)
varImpPlot(fit.rf2)
#Comparing the results
results<-resamples(list(GLM=fit.glm,KNN=fit.knn,NB=fit.nb,RPART=fit.rpart,SVM=fit.svm,RF=fit.rf))
summary(results)
dotplot(results)
#prediction and confusion matrix
pred.rf<-predict(fit.rf, test_set)
confusionMatrix(pred.rf, test_set$Loan_Status)
#Evaluation(by centering, scaling and boxcox transformation)
set.seed(99)
fit.glm<-train(Loan_Status~., train_set, method = "glm", metric = "Accuracy", preProc=c("center","scale","BoxCox"),trControl=control)
set.seed(99)
fit.knn<-train(Loan_Status~., train_set, method = "knn", metric = "Accuracy", preProc=c("center","scale","BoxCox"),trControl=control)
set.seed(99)
fit.nb<-train(Loan_Status~., train_set, method = "nb", metric = "Accuracy", preProc=c("center","scale","BoxCox"),trControl=control)
set.seed(99)
fit.rpart<-train(Loan_Status~., train_set, method = "rpart", metric = "Accuracy", preProc=c("center","scale","BoxCox"),trControl=control)
set.seed(99)
fit.svm<-train(Loan_Status~., train_set, method = "svmRadial", metric = "Accuracy", preProc=c("center","scale","BoxCox"),trControl=control)
set.seed(99)
fit.rf<-train(Loan_Status~., train_set, method = "rf", metric = "Accuracy", preProc=c("center","scale","BoxCox"),trControl=control)
#Comparing the algos
results<-resamples(list(GLM=fit.glm,KNN=fit.knn,NB=fit.nb,RPART=fit.rpart,SVM=fit.svm,RF=fit.rf))
summary(results)
dotplot(results)

#Tunning
grid<-expand.grid(.mtry=sqrt(col(train_set)))
#we could also use tuneLength parameter
set.seed(999)
fit.rf<-train(Loan_Status~., train_set, method = "rf", metric = "Accuracy", preProc=c("center","scale","BoxCox"),tuneGrid=grid,trControl=control)
fit.rf<-train(Loan_Status~., train_set, method = "rf", metric = "Accuracy", preProc=c("center","scale","BoxCox"),tuneLength=10,trControl=control)

set.seed(999)
grid2<-expand.grid(.sigma=c(0.025,0.05,0.1,0.15),.c=seq(1,10,by=1))
fit.svm<-train(Loan_Status~., train_set, method = "svmRadial", metric = "Accuracy", preProc=c("center","scale","BoxCox"),tuneGrid=grid2,trControl=control)

set.seed(999)
grid3<-expand.grid(.cp=c(0,0.05,0.1))
fit.rpart<-train(Loan_Status~., train_set, method = "rpart", metric = "Accuracy", preProc=c("center","scale","BoxCox"),tuneGrid = grid3, trControl=control)

set.seed(999)
grid4<-expand.grid(.k=seq(1,20,by=1))
fit.knn<-train(Loan_Status~., train_set, method = "knn", metric = "Accuracy", preProc=c("center","scale","BoxCox"),tuneGrid=grid4,trControl=control)
