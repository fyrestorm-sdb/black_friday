library(ggplot2)
library(caTools)
library(rpart)
library(rpart.plot)
library(data.table)
library(randomForest)
library(gbm)
library(dummies)

set.seed(123)

train=read.csv('train.csv')
#test=read.csv('test.csv') # non utilise ici...

str(train)
#'data.frame':   550068 obs. of  12 variables:
# $ User_ID                   : int  1000001 1000001 1000001 1000001 1000002 1000003 1000004 1000004 1000004 1000005 ...
# $ Product_ID                : Factor w/ 3631 levels "P00000142","P00000242",..: 673 2377 853 829 2735 1832 1746 3321 3605 2632 ...
 # $ Gender                    : Factor w/ 2 levels "F","M": 1 1 1 1 2 2 2 2 2 2 ...
 # $ Age                       : Factor w/ 7 levels "0-17","18-25",..: 1 1 1 1 7 3 5 5 5 3 ...
 # $ Occupation                : int  10 10 10 10 16 15 7 7 7 20 ...
 # $ City_Category             : Factor w/ 3 levels "A","B","C": 1 1 1 1 3 1 2 2 2 1 ...
 # $ Stay_In_Current_City_Years: Factor w/ 5 levels "0","1","2","3",..: 3 3 3 3 5 4 3 3 3 2 ...
 # $ Marital_Status            : int  0 0 0 0 0 0 1 1 1 1 ...
 # $ Product_Category_1        : int  3 1 12 12 8 1 1 1 1 8 ...
 # $ Product_Category_2        : int  NA 6 NA 14 NA 2 8 15 16 NA ...
 # $ Product_Category_3        : int  NA 14 NA NA NA NA 17 NA NA NA ...
 # $ Purchase                  : int  8370 15200 1422 1057 7969 15227 19215 15854 15686 7871 ...

 
# Product_ID
 ggplot(train,aes(x= Product_ID ))+geom_bar(fill="#FF9999", colour="black")+scale_fill_brewer(palette = "Pastel1") +theme(axis.title.x=element_blank(),axis.text.x=element_blank(),axis.ticks.x=element_blank())
 
 
# Gender proportion
round(prop.table(table(train$Gender))*100)  # proportion
# F  M 
#25 75

#Age proportion
round(prop.table(table(train$Age))*100)  # proportion
#0-17 18-25 26-35 36-45 46-50 51-55   55+ 
#    3    18    40    20     8     7     4 
ggplot(train,aes(x=Age))+geom_bar(aes(fill=train$Gender), colour="black", position=position_dodge())+scale_fill_brewer(palette = "Pastel1")+theme(axis.text.x =element_text(,hjust = 1,size=10))+ labs(fill = "Gender")

# Occupation
ggplot(train,aes(x=Occupation))+geom_bar(fill="#FF9999", colour="black")+scale_fill_brewer(palette = "Pastel1")+theme(axis.text.x =element_text(,hjust = 1,size=10))


# City category
View(train$City_Category)
round(prop.table(table(train$City_Category ))*100)  # proportion
# A  B  C 
#27 42 31 

#Stay_In_Current_City_Years
ggplot(train,aes(x=Stay_In_Current_City_Years))+geom_bar(fill="#FF9999", colour="black")+scale_fill_brewer(palette = "Pastel1")+theme(axis.text.x =element_text(,hjust = 1,size=10))

#Marital_Status
round(prop.table(table(train$Marital_Status ))*100)  # proportion
 #0  1 
#59 41 

# product category
ggplot(train,aes(x=Product_Category_1 ))+geom_bar(fill="#FF9994", colour="black")+scale_fill_brewer(palette = "Pastel1")+theme(axis.text.x =element_text(,hjust = 1,size=10))

# Purchase
 summary(train$Purchase)
 #  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
 #    12    5823    8047    9264   12054   23961 
 
 # which variable to sample ??
 task = makeClassifTask(data = train, target = "Purchase")
 var_imp <- generateFilterValuesData(task, method = c("information.gain"))
 plotFilterValues(var_imp,feat.type.cols = TRUE)
 var_imp
 # var_imp$data[order(-var_imp$data$information.gain),]
                         # name    type information.gain
# 2                  Product_ID  factor       3.67946269
# 11         Product_Category_3 integer       0.13878576
# 10         Product_Category_2 integer       0.12527707
# 4                         Age  factor       0.10205795
# 7  Stay_In_Current_City_Years  factor       0.06971162
# 6               City_Category  factor       0.04044840
# 3                      Gender  factor       0.02388183
# 1                     User_ID integer       0.00000000
# 5                  Occupation integer       0.00000000
# 8              Marital_Status integer       0.00000000
# 9          Product_Category_1 integer       0.00000000

#------>>> 2    Product_ID  factor       3.67946269
 
 
 #equilibrage ?
 #non, il n'y a pas une "classe" ultra majoritaire dans Product_ID
 
 

 
#####partitonnement train/set
 
sample=sample.split(train$Product_ID,SplitRatio=0.7)   
strain=train[sample==TRUE,]
stest=train[sample==FALSE,]

######## feature engineering
# convertir toutes les colonnes selectionnees en numeric
factcols <- c(2:4,6:7)

setDT(strain)[,(factcols) := lapply(.SD, as.numeric), .SDcols = factcols]  
setDT(stest)[,(factcols) := lapply(.SD, as.numeric), .SDcols = factcols]  


############regression lineaire
lm=lm(Purchase ~., strain)
summary(lm)

#Multiple R-squared:  0.1792,	Adjusted R-squared:  0.1792 
# F-statistic:  2318 on 11 and 116758 DF,  p-value: < 2.2e-16

lm_pred=predict(lm, newdata = stest)
lm_pred=data.frame(lm_pred)
table(is.na(lm_pred))
# FALSE   TRUE 
# 50051 114960 


############## arbre
tr=rpart(Purchase ~.,data=strain)
trpred=predict(tr, newdata = stest)
trpred=data.frame(trpred)
tree.error=with(stest,apply( (trpred-Purchase)^2,2,mean))  # rmse
sqrt(tree.error)
#  trpred 
#3102.995






########randomforest

randomfit <- randomForest(Purchase ~ User_ID+Product_ID+Gender+Age+Occupation+City_Category+Stay_In_Current_City_Years+Marital_Status+Product_Category_1 , data=strain,  importance=TRUE,  ntree=20)
randompred<- predict(randomfit, stest)
randompred=data.frame(randompred)
randomtree.error=with(stest,apply( (randompred-Purchase)^2,2,mean))  # rmse
sqrt(randomtree.error)
#2776.602


  
  
######### gradient boost

boost=gbm(Purchase ~ . ,data = strain,distribution = "gaussian",n.trees = 1000,shrinkage = 0.01, interaction.depth = 4)
summary(boost)
                                                 # var     rel.inf
# Product_Category_1                 Product_Category_1 88.89305507
# Product_ID                                 Product_ID  6.89390391
# Product_Category_2                 Product_Category_2  1.49762778
# User_ID                                       User_ID  1.05734118
# Product_Category_3                 Product_Category_3  0.82743411
# City_Category                           City_Category  0.36147148
# Occupation                                 Occupation  0.17869190
# Age                                               Age  0.16516855
# Stay_In_Current_City_Years Stay_In_Current_City_Years  0.06920978
# Gender                                         Gender  0.03359666
# Marital_Status                         Marital_Status  0.02249958

boostpred<-predict(boost,stest,n.trees = 1000)
boostpred=data.frame(boostpred)
boostpred.error=with(stest,apply( (boostpred-Purchase)^2,2,mean))  # rmse
sqrt(boostpred.error)
#boostpred 
#2974.267 

boost=gbm(Purchase ~ . ,data = strain,distribution = "gaussian",n.trees = 2000,shrinkage = 0.01, interaction.depth = 4)
boostpred<-predict(boost,stest,n.trees = 2000)
boostpred=data.frame(boostpred)
boostpred.error=with(stest,apply( (boostpred-Purchase)^2,2,mean))  # rmse
sqrt(boostpred.error)
#boostpred 
#2896.588 
  
boost=gbm(Purchase ~ . ,data = strain,distribution = "gaussian",n.trees = 10000,shrinkage = 0.01, interaction.depth = 4)
boostpred<-predict(boost,stest,n.trees = 10000)
boostpred=data.frame(boostpred)
boostpred.error=with(stest,apply( (boostpred-Purchase)^2,2,mean))  # rmse
sqrt(boostpred.error)
#boostpred 
#2738.975



############ F. engineering 2: age + gradient boost
ctrain=strain
ctrain$Age=as.integer(ctrain$Age )
ctrain$Age[ctrain$Age == 1] <- 15
ctrain$Age[ctrain$Age == 2] <- 21
ctrain$Age[ctrain$Age == 3] <- 30
ctrain$Age[ctrain$Age == 4] <- 40
ctrain$Age[ctrain$Age == 5] <- 48
ctrain$Age[ctrain$Age == 6] <- 53
ctrain$Age[ctrain$Age == 7] <- 60

ctest=stest
ctest$Age=as.integer(ctest$Age )
ctest$Age[ctest$Age == 1] <- 15
ctest$Age[ctest$Age == 2] <- 21
ctest$Age[ctest$Age == 3] <- 30
ctest$Age[ctest$Age == 4] <- 40
ctest$Age[ctest$Age == 5] <- 48
ctest$Age[ctest$Age == 6] <- 53
ctest$Age[ctest$Age == 7] <- 60

 
boost2=gbm(Purchase ~ . ,data = ctrain,distribution = "gaussian",n.trees = 10000,shrinkage = 0.01, interaction.depth = 4)
boostpred<-predict(boost2,ctest,n.trees = 10000)
boostpred=data.frame(boostpred)
boostpred.error=with(stest,apply( (boostpred-Purchase)^2,2,mean))  # rmse
sqrt(boostpred.error)
#boostpred 
#2738.975

  
  
######### plus de F. engineering dont remplacer product_id par les moyennes de Purchase
 


products=aggregate(ctrain, by=list(ctrain$Product_ID), FUN=mean)
mean_products=aggregate(ctrain, by=list(ctrain$Product_ID), FUN=mean)$Purchase
stddev_products=aggregate(ctrain, by=list(ctrain$Product_ID), FUN=sd)$Purchase
count_products=aggregate(ctrain, by=list(ctrain$Product_ID), FUN=length)$Purchase
ms_df=data.frame(products$Product_ID,mean_products, stddev_products,count_products)
colnames(ms_df)[1]="Product_ID"
last_train <- merge(ctrain, ms_df, by="Product_ID")
#last_train <- join(ctrain, ms_df, by="Product_ID")#préserve l'ordre des lignes

##### moyenne de Purchase vs son ecart type
ggplot(ms_df,aes(x=mean_products,y=stddev_products))+geom_point(size=1,aes(colour =count_products ))+scale_fill_brewer(palette = "Pastel1")+geom_abline(intercept = 0)




products=aggregate(ctest, by=list(ctest$Product_ID), FUN=mean)
mean_products=aggregate(ctest, by=list(ctest$Product_ID), FUN=mean)$Purchase
stddev_products=aggregate(ctest, by=list(ctest$Product_ID), FUN=sd)$Purchase
count_products=aggregate(ctest, by=list(ctest$Product_ID), FUN=length)$Purchase
ms_df=data.frame(products$Product_ID,mean_products, stddev_products,count_products)
colnames(ms_df)[1]="Product_ID"
last_test <- merge(ctest, ms_df, by="Product_ID")
#last_test <- join(ctest, ms_df, by="Product_ID") #préserve l'ordre des lignes



boost3=gbm(Purchase ~ .-Product_ID ,data = last_train,distribution = "gaussian",n.trees = 1000,shrinkage = 0.01, interaction.depth = 4)
boostpred<-predict(boost3,last_test,n.trees = 1000)
boostpred=data.frame(boostpred)
boostpred.error=with(last_test,apply( (boostpred-Purchase)^2,2,mean))  # rms
sqrt(boostpred.error)
#boostpred 
#2617.413 
 
summary(boost3)
                                                  # var      rel.inf
# mean_products                           mean_products 9.946696e+01
# User_ID                                       User_ID 1.686427e-01
# City_Category                           City_Category 1.352625e-01
# Occupation                                 Occupation 7.816469e-02
# stddev_products                       stddev_products 6.638487e-02
# Gender                                         Gender 2.687553e-02
# Age                                               Age 2.573249e-02
# Product_Category_1                 Product_Category_1 1.167520e-02
# Stay_In_Current_City_Years Stay_In_Current_City_Years 1.113865e-02
# count_products                         count_products 4.466559e-03
# Marital_Status                         Marital_Status 3.403313e-03
# Product_Category_2                 Product_Category_2 7.462508e-04
# Product_Category_3                 Product_Category_3 5.458136e-04
 
 
boost4=gbm(Purchase ~ . ,data = last_train,distribution = "gaussian",n.trees = 1000,shrinkage = 0.01, interaction.depth = 4)
boostpred<-predict(boost4,last_test,n.trees = 1000)
boostpred=data.frame(boostpred)
boostpred.error=with(last_test,apply( (boostpred-Purchase)^2,2,mean))  # rms
sqrt(boostpred.error)
#boostpred 
#2617.162 
 
boost5=gbm(Purchase ~ .-Product_ID ,data = last_train,distribution = "gaussian",n.trees = 2000,shrinkage = 0.01, interaction.depth = 4)
boostpred5<-predict(boost5,last_test,n.trees = 2000)
boostpred5=data.frame(boostpred5)
boostpred5.error=with(last_test,apply( (boostpred5-Purchase)^2,2,mean))  # rms
sqrt(boostpred5.error)
#boostpred5 
#2606.802
 
boost6=gbm(Purchase ~ .-Product_ID ,data = last_train,distribution = "gaussian",n.trees = 5000,shrinkage = 0.01interaction.depth = 4)
boostpred6<-predict(boost6,last_test,n.trees = 5000)
boostpred6=data.frame(boostpred6)
boostpred6.error=with(last_test,apply( (boostpred6-Purchase)^2,2,mean))  # rms
sqrt(boostpred6.error)
#boostpred6
#2590.733

boost7=gbm(Purchase ~ .-Product_ID ,data = last_train,distribution = "gaussian",n.trees = 10000,shrinkage = 0.01, interaction.depth = 4)
boostpred7<-predict(boost7,last_test,n.trees = 10000)
boostpred7=data.frame(boostpred7)
boostpred7.error=with(last_test,apply( (boostpred7-Purchase)^2,2,mean))  # rms
sqrt(boostpred7.error)
#boostpred7 
#2577.158
 
 
 ######export
 
 
write.csv(last_test[,c(1:13)],"xtest.csv",row.names = F)
write.csv(last_train[,c(1:13)],"xtrain.csv",row.names = F)
 
 
 
 
 
 train=read.csv('train.csv')
 otrain=train
 factcols <- c(2:4,6:7)
setDT(otrain)[,(factcols) := lapply(.SD, as.numeric), .SDcols = factcols]  
otrain=dummy.data.frame(otrain, names=c("City_Category"), sep="_")
otrain$Age=as.integer(otrain$Age )
otrain$Age[otrain$Age == 1] <- 15
otrain$Age[otrain$Age == 2] <- 21
otrain$Age[otrain$Age == 3] <- 30
otrain$Age[otrain$Age == 4] <- 40
otrain$Age[otrain$Age == 5] <- 48
otrain$Age[otrain$Age == 6] <- 53
otrain$Age[otrain$Age == 7] <- 60
otrain$Stay_In_Current_City_Years[otrain$Stay_In_Current_City_Years == 5] <- 4
sample = sample.split(otrain$User_ID, SplitRatio = 0.70)

ootrain=otrain[sample==TRUE,]
 ootest=otrain[sample==FALSE,]
 
 ootrain$User_ID=NULL
 ootest$User_ID=NULL
 
 product_mean <- ddply(ootrain, .(Product_ID), summarize, Product_Mean=mean(Purchase))
 ootrain <- merge(ootrain,product_mean, by="Product_ID")
 
  product_mean <- ddply(ootest, .(Product_ID), summarize, Product_Mean=mean(Purchase))
 ootest <- merge(ootest,product_mean, by="Product_ID")
 
# ootrain$Product_ID=NULL
  
 #ootest$Product_ID=NULL
 
 write.csv(ootest,'otest.csv')
 write.csv(ootrain,'otrain.csv')

 
 boostx=gbm(Purchase ~ . ,data = strain,distribution = "gaussian",n.trees = 1000,shrinkage = 0.01, interaction.depth = 4)
boostpredx<-predict(boostx,ctest,n.trees = 10000)
boostpredx=data.frame(boostpredx)
boostpredx.error=with(stest,apply( (boostpredx-Purchase)^2,2,mean))  # rmse
sqrt(boostpredx.error)
                