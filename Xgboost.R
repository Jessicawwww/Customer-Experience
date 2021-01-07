library(data.table)
library(fasttime)
library(corrplot)
library(moments)
library(glmnet)
library(randomForest)
library(caret)
library(ROSE)
library(gridExtra)
library(reshape2)
library(tidyverse)
library(DataExplorer)
library(dplyr)
library(plyr)
library(xgboost)
options(repr.matrix.max.rows=600, repr.matrix.max.cols=200)
setwd('C:/Users/lubai/Documents/Netlift Model/Netlift Model collection')

load(file = "April/c2c_t_train.RData")#c2c_t_train 128829
load(file = "April/c2c_t_test.RData")#c2c_t_test 55210
load(file = "April/c2c_c_train.RData")#c2c_t_train  10320
load(file = "April/c2c_c_test.RData")#c2c_t_test 4419

##################################################################################
#                                                                                #
#                               For Listing ind                                  #
#               **********************************************                   #
#                       whether the seller list or not                           #                                                                          #
##################################################################################

##################################################################################
                 #########2) Optin Effect Model#######
##################################################################################

select_data_col<-function(c2c,Yclass)
{
  c2c$ol = 0
  c2c$ol[c2c$ promo_optin_ind ==1 & c2c$promo_lstg_ind == 0] = 1
  c2c$ol[c2c$ promo_optin_ind ==0 & c2c$promo_lstg_ind == 1] = 2
  c2c$ol[c2c$ promo_optin_ind ==1 & c2c$promo_lstg_ind == 1] = 3
  exc_col = c('promo_dt','user_id','segment','max_t365_shop_level','promo_lstg_cnt',
              'ol','promo_lstg_ind','promo_optin_ind','tc_ind')
  selected = c2c[,-which(colnames(c2c) %in% exc_col)]
  return(list(X = as.matrix(selected),Y = as.numeric(c2c$ol==Yclass)))#return train data
}

for(i in 0:3)
{
  Trainlist_t = select_data_col(c2c_t_train,i)
  Testlist_t = select_data_col(c2c_t_test,i)
  
  # c2c_listing_train = ovun.sample(Trainlist_t$Y~., data = data.frame(Trainlist_t$X),method = 'over')$data
  #x_train = as.matrix(c2c_listing_train[,-ncol(c2c_listing_train)])
  dtrain_class <- xgb.DMatrix(Trainlist_t$X, label = Trainlist_t$Y)
  dtest_class <-xgb.DMatrix(Testlist_t$X, label = Testlist_t$Y)
  watchlist_class <- list(train = dtrain_class, eval = dtest_class)
  ## A simple xgb.train example:
  param_class <- list(max_depth = 3, eta = 0.2, nthread = 2,
                      objective = "binary:logistic", eval_metric = "auc")
  bst_class <- xgb.train(param_class, dtrain_class, nrounds =100, watchlist_class)
  save(bst_class,file=paste("xgb_Y",i,".RData",sep=""))
  #auc
  #Y=0 0.895
  #Y=1 0.806
  #Y=2 0.870
  #Y=3 0.927
}



##################################################################################
                          #########3) T_learner#######
################################################################################

select_data_col_Tlearner<-function(c2c)
{
  exc_col = c('promo_dt','user_id','segment','max_t365_shop_level','promo_lstg_cnt',
              'promo_lstg_ind','promo_optin_ind','tc_ind')
  selected = c2c[,-which(colnames(c2c) %in% exc_col)]
  return(list(X = as.matrix(selected),Y = c2c$promo_lstg_ind))#return train data
}

xgbtrain<-function(train,test,selection_method,max_depth = 3, eta = 0.2, nrounds = 50)
{
  Trainlist_t = selection_method(train)
  Testlist_t = selection_method(test)
  
  dtrain_class <- xgb.DMatrix(Trainlist_t$X, label =Trainlist_t$Y)
  dtest_class <-xgb.DMatrix(Testlist_t$X, label = Testlist_t$Y)
  watchlist_class <- list(train = dtrain_class, eval = dtest_class)
  ## A simple xgb.train example:
  param_class <- list(max_depth = max_depth, eta = eta, nthread = 2,
                      objective = "binary:logistic", eval_metric = "auc")
  bst <- xgb.train(param_class, dtrain_class, nrounds =nrounds, watchlist_class)
  return(bst)
}

bst_Tlearn_t = xgbtrain(c2c_t_train,c2c_t_test,select_data_col_Tlearner)
#auc: 0.903
save(bst_Tlearn_t,file="xgb_bst_Tlearn_t.RData")
bst_Tlearn_c = xgbtrain(c2c_c_train,c2c_c_test,select_data_col_Tlearner)
#auc: 0.900
save(bst_Tlearn_c,file="xgb_bst_Tlearn_c.RData")

##################################################################################
                        #########4) S_learner#######
################################################################################
select_data_col_Slearner<-function(c2c)
{
  #Slearner uses tc_ind indicator as a predictor
  exc_col = c('promo_dt','user_id','segment','max_t365_shop_level','promo_lstg_cnt',
              'promo_lstg_ind','promo_optin_ind')
  selected = c2c[,-which(colnames(c2c) %in% exc_col)]
  return(list(X = as.matrix(selected),Y = c2c$promo_lstg_ind))#return train data
}
bst_Slearn = xgbtrain(rbind(c2c_t_train,c2c_c_train),
                      rbind(c2c_t_test,c2c_c_test),
                      select_data_col_Slearner)
#auc: 0.903
save(bst_Slearn,file="xgb_bst_Slearn.RData")


########################causal forest#############
library(uplift)
cfdata_train = select_data_col_Slearner(rbind(c2c_t_train,c2c_c_train))
cfdata_train_frame = as.data.frame(cbind(cfdata_train$Y,cfdata_train$X))
fit1 <- ccif(formula =V1~trt(tc_ind)+.,
             data = cfdata_train_frame, 
             ntree = 50, 
             split_method = "Int",
             distribution = approximate (B=999),
             pvalue = 0.05,
             verbose = TRUE)
##################################################################################
                        #########5) X_learner#######
################################################################################

load(file="xgb_bst_Tlearn_c.RData") #bst_Tlearn_c
load(file="xgb_bst_Tlearn_t.RData") #bst_Tlearn_t


select_data_col_Xlearner<-function(c2c,model_t,model_c,type = "t")
{
  #Slearner uses tc_ind indicator as a predictor
  exc_col = c('promo_dt','user_id','segment','max_t365_shop_level','promo_lstg_cnt',
              'promo_lstg_ind','promo_optin_ind','tc_ind')
  selected = c2c[,-which(colnames(c2c) %in% exc_col)]
  selected =  as.matrix(selected)
  if(type == "t")
  {
    c2c$treatment = c2c$promo_lstg_ind  -  predict(model_c,selected)
  }
  if(type == "c")
  {
    c2c$treatment = predict(model_t,selected) - c2c$promo_lstg_ind
  }
  return(list(X = selected,Y = c2c$treatment))#return train data
}


xgb_xlearner<-function(train,test,selection_method,
                       model_t,model_c,type = "t",
                       max_depth = 3, eta = 0.2, nrounds = 50)
{
  Trainlist  = selection_method(train,model_t,model_c,type= type)
  Testlist  = selection_method(test,model_t,model_c,type= type)
  
  dtrain_class <- xgb.DMatrix(Trainlist$X, label =Trainlist$Y)
  dtest_class <-xgb.DMatrix(Testlist$X, label = Testlist$Y)
  watchlist_class <- list(train = dtrain_class, eval = dtest_class)
  ## A simple xgb.train example:
  param_class <- list(max_depth = max_depth, eta = eta, nthread = 2,
                      objective = "reg:squarederror", eval_metric = "rmse")
  bst <- xgb.train(param_class, dtrain_class, nrounds =nrounds, watchlist_class)
  return(bst)
}
bst_xgb_xlearner_t = xgb_xlearner(c2c_t_train,c2c_t_test,select_data_col_Xlearner,
                              bst_Tlearn_t,bst_Tlearn_c,type="t")
bst_xgb_xlearner_c = xgb_xlearner(c2c_c_train,c2c_c_test,select_data_col_Xlearner,
                                  bst_Tlearn_t,bst_Tlearn_c,type="c",
                                  max_depth = 2,eta = 0.1,nrounds = 50)
save(bst_xgb_xlearner_t,file="xgb_bst_Xlearner_t.RData")
save(bst_xgb_xlearner_c,file="xgb_bst_Xlearner_c.RData")

