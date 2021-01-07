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
options(repr.matrix.max.rows=600000, repr.matrix.max.cols=2000)

# Loading  dataset
all_data = read.table(file = 'C:/Users/weiwewu/eBay Inc/AU Seller Project - Documents/6_Customer Experience Survey/CX_Monitor.txt', header = T)

head(c2c)
str(c2c)

exc_col = c('promo_dt','user_id','segment', 'promo_optin_ind' , 'max_t365_shop_level')

c2c = c2c[,-which(colnames(c2c) %in% exc_col)]
head(c2c)
str(c2c)

# Histogram of the response variable promo_lstg_ind
ggplot(c2c, aes(x = promo_lstg_ind)) + geom_histogram(fill = "lightblue") 

## The response is not so unbalanced, so no need to balance it


# Correlation analysis
c2c[, -which(colnames(c2c) %in% c('promo_lstg_ind'))] %>% 
  map(function(x) cor(c2c[,'promo_lstg_ind'], x)) %>% 
  as.data.frame %>% 
  gather %>% 
  mutate(correlation = ifelse(abs(value) > 0.3, "medium correlation", "low correlation")) %>% 
  ggplot(aes(x = reorder(key, value), y = value)) + geom_point(aes(color = correlation)) + coord_flip() + ylab("correlation with promo_lstg_ind") + theme(text = element_text(size = 7))


correlation = cor(c2c[, -which(colnames(c2c) %in% c('promo_lstg_ind'))] , c2c$promo_lstg_ind)
correlation = as.data.frame(as.table(correlation))
head(correlation[with(correlation, order(Freq)),][,c('Var1','Freq')], n=10)

tail(correlation[with(correlation, order(Freq)),][,c('Var1','Freq')],n=10)


# Correlations of the feature promo_lstg_ind with all the other features in c2c sorted by absolute value
correlations = map_dbl(c2c[, -which(colnames(c2c) %in% c('promo_lstg_ind'))] , function(x) cor(c2c[, 'promo_lstg_ind'], x))
correlations = sort(abs(correlations))
correlations

#Choosing the top 65 features by "importance" as a consequence of the previous sorting
features = names(correlations)[66:130]
all_selected = c('promo_lstg_ind',features)
selected = c2c[ , all_selected]
head(selected)
str(selected)
plot_correlation(selected) 

for(i in 1:ncol(selected)){
  
  cat("\n","Range of values of", all_selected[i],"\n","\n")
  
  print(range(selected[[all_selected[i]]]))
  
}

# Log Transform

c2c %>% 
  map(function(x) skewness(x)) %>% 
  as.data.frame %>% 
  gather %>% 
  mutate(skewed = ifelse(abs(value) > 1, "skewed", "symmetric")) %>% 
  ggplot(aes(x = reorder(key, value), y = value))+geom_point(aes(color = skewed))+coord_flip()+ylab("skewness")+ theme(text = element_text(size=7))  

# Skeweness and histogram of t180_gmv_rate before log transformation (example)
skewness(c2c$t180_gmv_rate) # 84.99896
ggplot(c2c, aes(x = t180_gmv_rate)) + geom_histogram(fill = "lightblue") 


# Log transformation of the right skewed features with range  >100
for(i in 1:ncol(c2c)){
  if(skewness(c2c[[i]])>1 & (max(c2c[[i]])-min(c2c[[i]]))>100){
    c2c[[i]]=log(c2c[[i]]+1-min(c2c[[i]])) # translation of the argument of the log to deal with negative values (and to have 1 as minimum)
  }
}

# Skeweness and histogram of t180_gmv_rate after log transformation (example)
skewness(c2c$t180_gmv_rate)  # 2.173724
ggplot(c2c, aes(x = t180_gmv_rate)) + geom_histogram(fill = "lightblue") 


## Since a nonlinear transformation alters (either increases or decreases) the linear relationships between variables, the log transform should have changed the correlation between the variables. We plot the correlation matrix of the features with the response after the log transform.
orrelations = map_dbl(c2c[, -which(colnames(c2c) %in% c('promo_lstg_ind'))],function(x) cor(c2c[, 'promo_lstg_ind'], x))
correlations = sort(abs(correlations))
features = names(correlations)[66:130]
all_selected = c('promo_lstg_ind',features)
selected = c2c[,all_selected]
plot_correlation(selected)
str(selected)

#If we focus on the row (column) promo_lstg_ind, that is our response variable, both before and after the log trasformation, it is clear that there are more features moderately correlated with the response variable after the log transform. Moreover, in general, among the  6565  features mostly correated with the response varaible, there are stronger correlations after the log transform.
# Anyway, also after the log transform, the correlations with the response variable are not so high to give some insights on what variable to choose in order to model our problem. Maybe the use of some machine learning tecnique could be of help.


# Feature distributions split by response variable
# How different looks like the feature distributions for the promo responders versus non-responders?

plot_boxplot(c2c,'promo_lstg_ind')



# Features selection

c2c_lasso = selected[, -which(colnames(selected) %in% c('promo_lstg_cnt'))]
trainIndex = createDataPartition(c2c_lasso$promo_lstg_ind, p = .7, list = FALSE)
c2c_lasso_train = c2c_lasso[trainIndex,]
c2c_lasso_test = c2c_lasso[-trainIndex,]
x_train = model.matrix(promo_lstg_ind~ .,c2c_lasso_train)
y_train = c2c_lasso_train$promo_lstg_ind
cv.out.lasso = cv.glmnet(x_train,y_train,alpha=1,family="binomial",type.measure ="class")
lambda_1se = cv.out.lasso$lambda.1se 
cat("\n","survived coefficients","\n", "\n")
coef(cv.out.lasso, s=lambda_1se)
x_test = model.matrix(promo_lstg_ind~ .,c2c_lasso_test)
lasso_prob = predict(cv.out.lasso, newx = x_test, type="response")
predlasso = ifelse(lasso_prob>.5, 1, 0)
cat("\n","confusion matrix","\n", "\n")
print(table(pred=predlasso,true=c2c_lasso_test$promo_lstg_ind))
cat("\n","accuracy","\n", "\n")
print(mean(predlasso==c2c_lasso_test$promo_lstg_ind))  # 0.8209124


# Coefficients output of the Lasso as data frame (and data frame with their absolute values), discarding the intercept
lasso_coef = as.data.frame(as.matrix(coef(cv.out.lasso, s=lambda_1se)))
names(lasso_coef)="Coef"

abs_lasso_coef = lasso_coef[3:nrow(lasso_coef),,drop=FALSE]
abs_lasso_coef$abs_Coef = abs(abs_lasso_coef$Coef)
abs_lasso_coef

survived_to_lasso = rownames(abs_lasso_coef)[abs_lasso_coef$abs_Coef != 0]
lengthsurv=length(survived_to_lasso)
lengthsurv  # 29


# Now we use logistic regression on the selected data set to predict promo_lstg_ind with only the features that survived to Lasso to understand which, among these variables, are truly statistically significant (significance code  ??????????????????????????? )
# c2c_log = selected
# c2c_log$prev_optin_cnt <- NULL

c2c_log = selected[, -which(colnames(selected) %in% c('promo_lstg_cnt'))]
promo_lstg_ind_log = glm(as.formula(paste("promo_lstg_ind~", paste(survived_to_lasso, collapse="+"))),family = "binomial",data=c2c_log)
summary(promo_lstg_ind_log)


c2c_log = selected[, -which(colnames(selected) %in% c('promo_lstg_cnt'))]
trainIndex = createDataPartition(c2c_log$promo_lstg_ind, p = .7, list = FALSE)
c2c_log_train = c2c_log[trainIndex,]
c2c_log_test = c2c_log[-trainIndex,]
new_survived_to_lasso = survived_to_lasso[!survived_to_lasso%in%c('t365_gmv_amt','t180_pickup_only_rate','t30_srch_cnt', 't90_fvf_rate', 't365_si_cnt')]
fit = glm(as.formula(paste("promo_lstg_ind~", paste(new_survived_to_lasso, collapse="+"))), 
          family = "binomial",c2c_log_train)
summary(fit)
log_prob = predict(fit,newdata=c2c_log_test,type='response')
log_pred = ifelse(log_prob > 0.5,1,0)
cat("\n","\n","accuracy","\n", "\n")
print(mean(log_pred==c2c_log_test$promo_lstg_ind))  # 0.8254412


# Random Forest
c2c_rf = selected[, -which(colnames(selected) %in% c('promo_lstg_cnt'))]
c2c_rf$promo_lstg_ind = factor(c2c_rf$promo_lstg_ind)
trainIndex = createDataPartition(c2c_rf$promo_lstg_ind, p = .7, list = FALSE)
c2c_rf_train = c2c_rf[trainIndex,]
c2c_rf_test = c2c_rf[-trainIndex,]
Random_Forest = randomForest(promo_lstg_ind ~ ., data=c2c_rf_train,ntree=50)
predrf = predict(Random_Forest, newdata = c2c_rf_test, importance=TRUE)
importance(Random_Forest, type=2)
cat("\n","Number of trees","\n", "\n")
print(Random_Forest$ntree)
cat("\n","Variable importance","\n", "\n")
varImpPlot(Random_Forest)
cat("\n","Confusion matrix","\n", "\n")
tbrf = table(predrf, c2c_rf_test$promo_lstg_ind)
print(tbrf)
cat("\n","Accuracy", "\n", "\n")
print(mean(predrf==c2c_rf_test$promo_lstg_ind))  # 0.823643

Importance = data.frame(importance(Random_Forest, type=2))
Importance[,'MeanDecreaseGini']=(Importance[,'MeanDecreaseGini']-min(Importance[,'MeanDecreaseGini']))/(max(Importance[,'MeanDecreaseGini'])-min(Importance[,'MeanDecreaseGini']))
names(Importance) = 'RF'

# Adding a column with the absolute values of the Lasso Coefficients (measure of variable importance for Lasso), converting to a data table and viewing the final Importance data tabl
Importance[,'Lasso'] = abs_lasso_coef$abs_Coef
Importance[,'Lasso']=(Importance[,'Lasso']-min(Importance[,'Lasso']))/(max(Importance[,'Lasso'])-min(Importance[,'Lasso']))
Importance = setDT(Importance,keep.rownames = TRUE)[]
setnames(Importance,1,'Variables')
Importance
#Plots of variables importance of Logistic Regression vs Random Forest
names(Importance) = c("Variables", "RF", "Lasso")
Importance.melt = melt(Importance, id.vars = 'Variables', variable.name = 'Algo', 
                       value.name = 'Importance')
ggplot(Importance.melt, aes(x = Variables, y = Importance, shape = Algo))+
  geom_point() + scale_shape_manual(values = c("RF" = 8,'Lasso' = 1)) + 
  coord_flip()

#Features to keep for future analysis as consequence of the Lasso and of the Random Foreset importance measures: the ones whose Lasso coefficient is not zero and the ones with Lasso coefficient zero but random forest importance greather or equal to  0.010.01
Importance=as.data.frame(Importance)
selected_features_1 = filter(Importance,Lasso!=0.0)
selected_features_2 = filter(Importance,Lasso==0.0, Importance$RF>=0.01)
selected_features = rbind(selected_features_1,selected_features_2)
selected_features_Lasso_RF = selected_features[,1]
cat(selected_features_Lasso_RF,sep="\n")
length(selected_features_Lasso_RF)  # 59


# For the moment we decide to keep for the future analysis:

# 1. the ones whose (global) Lasso coefficient is not zero and the ones with (global) Lasso coefficient zero but random forest importance greather or equal to  0.010.01 
# 2. some features that are not between the  6565  more correlated to the response variable but that could have some importance for the future analysis anyway (i.e. t90_email_or,t30_email_or,t90_email_ctor,t30_email_ctor)
selected_features=c(
  't90_email_or',
  't30_email_or',
  't90_email_ctor',
  selected_features_Lasso_RF)
selected_features
length(selected_features)  # 62

# In order to check if the features that are not between the  6565  more correlated to the response (t90_email_or,t30_email_or,t90_email_ctor,t30_email_ctor) are worth to be included in the final vector of selected_features, we run a Lasso regression to predict the response using only the selected_features to check how many of them will survive.
features = c('promo_lstg_ind',selected_features)
c2c_lasso_final = c2c[,features]
str(c2c_lasso_final)

trainIndex = createDataPartition(c2c_lasso_final$promo_lstg_ind, p = .7, list = FALSE)
c2c_lasso_final_train = c2c_lasso_final[trainIndex,]
c2c_lasso_final_test = c2c_lasso_final[-trainIndex,]
x_train = model.matrix(promo_lstg_ind~ .,c2c_lasso_final_train)
y_train = c2c_lasso_final_train$promo_lstg_ind
cv.out.lasso.final = cv.glmnet(x_train,y_train,alpha=1,family="binomial",type.measure ="class")
lambda_1se = cv.out.lasso.final$lambda.1se 
cat("\n","final survived coefficients","\n", "\n")
coef(cv.out.lasso.final, s=lambda_1se)
x_test = model.matrix(promo_lstg_ind~ .,c2c_lasso_final_test)
lasso_prob_final = predict(cv.out.lasso.final, newx = x_test, type="response")
predlassofinal = ifelse(lasso_prob_final>.5, 1, 0)
cat("\n","confusion matrix","\n", "\n")
print(table(pred=predlassofinal,true=c2c_lasso_final_test$promo_lstg_ind))
cat("\n","accuracy","\n", "\n")
print(mean(predlassofinal==c2c_lasso_final_test$promo_lstg_ind))  # 0.825641
lasso_final_coef = as.data.frame(as.matrix(coef(cv.out.lasso.final, s=lambda_1se)))
names(lasso_final_coef)="Coef"
lasso_final_coef = lasso_final_coef[3:nrow(lasso_final_coef),,drop=FALSE]
survived_to_final_lasso = rownames(lasso_final_coef)[lasso_final_coef$Coef != 0]
cat("\n","Number of features survived to the final Lasso","\n", "\n")
lengthfinalsurv=length(survived_to_final_lasso)  # 36
lengthfinalsurv
cat("\n","Features survived to the final Lasso","\n", "\n")
cat(survived_to_final_lasso,sep="\n")

# There are features, among the ones that are not between the  6565  more correlated to the response (t30_email_or,t30_email_ctor), that survived to the final Lasso.
# We decide to keep only the features that survived from this last Lasso for the future analysis.

columns = c('promo_lstg_ind',survived_to_final_lasso)
c2c_final = c2c[,columns]
write.csv(c2c_final,file="/home/weiwewu/Uplift Model/AU/c2c_final.csv")
str(c2c_final)
