library(dplyr)
library(car)
library(caret)
library(randomForest)
library(cvTools)
library(pROC)
#createdummnies function
source("createDummies.r")

#Impoting data in R
bank_train = read.csv("bank-full_train.csv")
bank_test = read.csv("bank-full_test.csv")

#Adding target variable "y" in test data
bank_test$y = NA

#Adding new column to differentiate between train and test data after combining
bank_test$data ="test"
bank_train$data = "train"

#Combining train and test data
all_bank = rbind(bank_train,bank_test)
glimpse(all_bank)

#Finding NA values
sum(is.na(all_bank))

#retriving columns with binary values ie yes and no
col_binary = c("housing", "loan","default", "y")

#converting binary columns to numeric
for (i in col_binary) {
  all_bank[i] = as.numeric(all_bank[i] == "yes")
}

#retriving categorical columns
cat_col = names(all_bank)[sapply(all_bank, function(x) is.character(x))];cat_col

#creating dummies for categorical columns
for (col in cat_col) {
  
  all_bank = createDummies(all_bank, col, 0)
  
}

#Splitting train and test data from combined data
bank_train = all_bank %>% filter(data_train == 1) %>% select(-data_train)
bank_test = all_bank %>% filter(data_train == 0) %>% select(-data_train,-y)


#Spliting train data 
s = sample(1:nrow(bank_train), 0.75 * nrow(bank_train))
train75 = bank_train[s,]
train25 = bank_train[-s,]
bank_train$y = as.factor(bank_train$y)

                                 
#Building Gradient Boosting Machine (GBM) model-----------------------------------------------
logistic = glm(y ~. -ID, data = train75, family = binomial); logistic
#removing columns with high vif and p-values
summary(logistic)
sort(vif(logistic), decreasing = T)[1:5]
                                 
#final model
logistic1 = glm(y ~. -ID -job_management-month_may-education_secondary-pdays-
                  job_unemployed-job_technician  -job_entrepreneur-marital_single-
                  default-age-previous-contact_cellular-job_self_employed-
                  job_services,
                data = train75, family = binomial)
round(sort((summary(logistic1)$coefficients)[,4]),2) %>% tail()
sort(vif(logistic1), decreasing = T)[1:5]

#predicting GBM on trained dataset
train_bank_score = predict(logistic1, newdata = train75, type = "response")
train_num = ifelse(train_bank_score>0.5, 1,0)
                                 
#predicting GBM on train25 dataset
test_bank_score = predict(logistic1, newdata = train25, type = "response")
test_num = ifelse(test_bank_score > 0.5, 1, 0)
                                 
#Area under curve
auc(roc(train25$y, test_num))

#predicting GBM on train dataset
bank_train_log = predict(logistic1, newdata = bank_train, type = "response")
predicted = ifelse(bank_train_log > 0.13, 1, 0)
cm_bank_train = confusionMatrix(factor(train_num), factor(train75$y))

#predicting GBM on test dataset
test.pred = predict(logistic1, newdata = bank_test, type = "response")
test.value = ifelse(test.pred > 0.13, 1, 0)


#Creating confusion matrix
cm_bank_test = confusionMatrix(factor(test_num),
                               factor(train25$y))

real = bank_train$y


TP = sum(real ==1 & predicted == 1);TP
TN = sum(real == 0 & predicted == 0);TN
FP =sum(real == 0 & predicted == 1); FP
FN =sum(real == 1 & predicted == 0);FN


P = TP + FN; P
N = TN + FP; N
KS = (TP/P) -(FP/N); KS
cutoffs <- seq(0.01, 0.99, 0.01); cutoffs
cutoff_data <- data.frame(cutoff = 99, Accuracy = 99, 
                          Sn = 99, Sp=99, KS=99, 
                          F5=99, F.1=99, M=99)
View(cutoff_data)

for(cutoff in cutoffs){
  
  predicted = as.numeric(bank_train_log > cutoff)
  
  TP=sum(real==1 & predicted==1)
  TN=sum(real==0 & predicted==0)
  FP=sum(real==0 & predicted==1)
  FN=sum(real==1 & predicted==0)
  
  P=TP+FN
  N=TN+FP
  
  Accuracy=(TP+TN)/(P+N)
  Sn = TP/P
  Sp = TN/N
  precision = TP/(TP + FP)
  recall = Sn
  
  KS = (TP/P) - (FP/N)
  F5 = (26*precision*recall)/((25*precision) + recall)
  F.1=(1.01*precision*recall)/((.01*precision)+recall)
  
  M = (4*FP+FN)/(5*(P+N))
  
  cutoff_data = rbind(cutoff_data,
                      c(cutoff,Accuracy, Sn,Sp,KS,F5,F.1,M))
}

cutoff_data = cutoff_data[-1,  ]
View(cutoff_data)










