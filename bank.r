library(dplyr)
library(car)
library(caret)
library(randomForest)
library(cvTools)
library(pROC)
setwd("C:/Users/ASUS/OneDrive/Desktop/Edvancer R/R")
source("createDummies.r")

setwd("C:/Users/ASUS/OneDrive/Desktop/Edvancer R/R/projectdb")

bank_train = read.csv("bank-full_train.csv")
bank_test = read.csv("bank-full_test.csv")
bank_test$y = NA
bank_test$data ="test"
bank_train$data = "train"
all_bank = rbind(bank_train,bank_test)
glimpse(all_bank)

sum(is.na(all_bank))
col_binary = c("housing", "loan","default", "y")

for (i in col_binary) {
  all_bank[i] = as.numeric(all_bank[i] == "yes")
}

cat_col = names(all_bank)[sapply(all_bank, function(x) is.character(x))];cat_col

for (col in cat_col) {
  
  all_bank = createDummies(all_bank, col, 0)
  
}

#-----------------------------------------------------------
bank_train = all_bank %>% filter(data_train == 1) %>% select(-data_train)
bank_test = all_bank %>% filter(data_train == 0) %>% select(-data_train,-y)

#------------------------------------------------------------

s = sample(1:nrow(bank_train), 0.75 * nrow(bank_train))
train75 = bank_train[s,]
train25 = bank_train[-s,]
bank_train$y = as.factor(bank_train$y)

logistic = glm(y ~. -ID, data = train75, family = binomial); logistic

summary(logistic)
sort(vif(logistic), decreasing = T)[1:5]

logistic1 = glm(y ~. -ID -job_management-month_may-education_secondary-pdays-
                  job_unemployed-job_technician  -job_entrepreneur-marital_single-
                  default-age-previous-contact_cellular-job_self_employed-
                  job_services,
                data = train75, family = binomial)
round(sort((summary(logistic1)$coefficients)[,4]),2) %>% tail()
sort(vif(logistic1), decreasing = T)[1:5]


train_bank_score = predict(logistic1, newdata = train75, type = "response")
train_num = ifelse(train_bank_score>0.5, 1,0)

test_bank_score = predict(logistic1, newdata = train25, type = "response")
test_num = ifelse(test_bank_score > 0.5, 1, 0)

auc(roc(train25$y, test_num))

bank_train_log = predict(logistic1, newdata = bank_train, type = "response")
predicted = ifelse(bank_train_log > 0.13, 1, 0)
cm_bank_train = confusionMatrix(factor(train_num), factor(train75$y))

test.pred = predict(logistic1, newdata = bank_test, type = "response")
test.value = ifelse(test.pred > 0.13, 1, 0)

write.table(test.value, "11july_bank.csv", row.names = F, col.names = "y")




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


#Find mean of the variable age. Round off to 2 decimal places.

round(mean(bank_train$age),2)

#Total number of outliers present in the variable balance.
#Use ‘Q1-1.5*IQR’ to calculate lower limit and ‘Q3 + 1.5×IQR’ 
#to calculate upper limit. calculate the count of 
#values in variable balance which are beyond these limits.


q1 = quantile(bank_train$balance, 0.25);q1
q2 = quantile(bank_train$balance, 0.75);q2
IQR = q2 -q1 ; IQR
lower_limit <- q1 - 1.5 * IQR
upper_limit <- q2 + 1.5 * IQR

outliers = sum(bank_train$balance < lower_limit| bank_train$balance > upper_limit)


var(bank_train$balance)









