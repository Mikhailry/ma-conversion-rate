# set dir
setwd('/Users/mikhailrybalchenko/Projects/Marketing cases/Conversion rate/')


#========libraries and data load============
# load libraries
library('tidyverse')
#install.packages('caret') #uncomment to install
library('caret')
#install.packages('randomForest')
library('randomForest')
library('ggplot2')
#install.packages('precrec')
library('precrec')
#install.packages('rpart')
library('rpart')

# load data
data <- read_csv('conversion_data.csv')

# let's identify numerical and categorical variables
numVars <- c('age', 'total_pages_visited')
catVars <- setdiff(variable.names(data), numVars)

# convert categoricals into factors
data$country <- as.factor(data$country)
data$new_user <- as.factor(data$new_user)
data$source <- as.factor(data$source)
#data$converted <- as.factor(data$converted)


#============+EDA+============================
# let's check how age and total pages visited are correlated
cor(data[,numVars])
# there is no correlation between age and total_pages_visited


# let's check summary statistics
summary(data)

# we will remove observation of user older than 80 years
data <- data %>% filter(age <= 80)
#data <- subset(data, age<80) # alternative way

# check for missing data column-wise
sapply(data, function(x) sum(is.na(x)))
# there is no missing data




# convertion by country
conv_by_country <- data %>% count(country, converted) 
conv_by_country_prop <- conv_by_country %>% group_by(country) %>% mutate(CR = n/sum(n)*100) %>% filter(converted==1)
conv_by_country_prop

# alternatively
data_country = data %>% group_by(country) %>% summarise(conversion_rate = mean(converted))

ggplot(data=data_country, aes(x=country, y=conversion_rate))+
  geom_bar(stat = "identity", aes(fill = country))
# China has the smallest CR = 0.13%, while Germany has the highest CR of 6.25% from traffic


# conversion rate by number of pages visited
data_pages = data %>% group_by(total_pages_visited) %>% summarise(conversion_rate = mean(converted))
qplot(total_pages_visited, conversion_rate, data=data_pages, geom="line")
# visiting more pages correlated with the conversion


# convertion by traffic source
conv_by_source <- data %>% count(source, converted) 
conv_by_source_prop <- conv_by_source %>% group_by(source) %>% mutate(CR = n/sum(n)*100) %>% filter(converted==1)
conv_by_source_prop
# highest conversion is coming from ADS traffic = 3.45%, then SEO = 3.29%. Given that Direct traffic CR is just a bit lower, it might be feasible to lower ADS budget.



# convertion by country and traffic source
conv_by_country_source <- data %>% count(country, source, converted) 
conv_by_country_source_prop <- conv_by_country_source %>% group_by(country, source) %>% mutate(CR = n/sum(n)*100) %>% filter(converted==1)
conv_by_country_source_prop
# from the table we can see which channels bring higher conversion in each country


# convertion by country and user new vs existing
conv_by_country_user <- data %>% count(country, new_user, converted) 
conv_by_country_user_prop <- conv_by_country_user %>% group_by(country, new_user) %>% mutate(CR = n/sum(n)*100) %>% filter(converted==1)
conv_by_country_user_prop
# from this table we can see that new_user's conversion is much lower than for existing users. We can think of some incentives for the new users to convert them faster


# mean age of converted users by country 
data_country_conv <- data %>% group_by(country, converted) %>% summarise(mean_age = mean(age))
data_country_conv
# on average, younger people convert better in all countries

# mean pages visited of converted users by country
data %>% group_by(country, converted) %>% summarise(mean_pages = mean(total_pages_visited))
# on average, converted users visit 14.5 pages, opposed to the not converted visiting 4.5 pages

#=======modeling========================

# change converted to factor
data$converted <- as.factor(data$converted)

# define target variable
target <- data$converted

#setting seed
set.seed(20191126)

#Doing stratified sampling
trainInd <- createDataPartition(target, p = 0.8, list = FALSE)
trainData <- data[trainInd,]
testData <- data[-trainInd,]
stopifnot(nrow(trainData) + nrow(testData) == nrow(data))


# let's fit the randomForest model 
data.rf <- randomForest(converted ~ ., data=trainData, ntree=150, keep.forest=TRUE)

# model summary
print(data.rf)

# OOB error is low -> model is not overfitted. 

# plot rf
plot(data.rf)


testData$pred <- predict(data.rf, testData)
testData$pred_plot <- testData$pred
testData$pred <- as.factor(testData$pred)
confusionMatrix(testData$pred, testData$converted)


# even though the accuracy is 98.5%, 
# Sensitivity (recall or true positive rate), percentage of true records that we predicted correctly, is 99.6% meaning that we correctly classified 99.6% of converted users
# Specificity (true negative rate), measures what portion of the actual false records we predicted correctly, is 68.4% meaning that 68.4% of users that were not converted, 
# were classified correctly as not converted, but 31.6% were classified as converted while they were not (false-positives).


rocCurve <- evalmod(scores = as.numeric(as.character(testData$pred)), labels = as.numeric(as.character(testData$converted)))
autoplot(rocCurve)


# Variable Importance by Information gain
varImpPlot(data.rf,  
           sort = T,
           n.var=5,
           main="Top 5 - Variable Importance")


# in terms of variable importance, total_pages_visited has the highest effect on conversion.
# unfortunately, we can't influence that variable too much, so let's rebulid the model without it.
trainData2 <- trainData
testData2 <- testData
trainData2$total_pages_visited <- NULL
testData2$total_pages_visited <- NULL



# let's fit the randomForest model2
# Since classes are heavily unbalanced and we don’t have that very powerful variable anymore, let’s change the weight a bit, 
# just to make sure we will get something classified as 1

data.rf2 <- randomForest(converted ~ ., data=trainData2, ntree=150, keep.forest=TRUE, classwt = c(0.7,0.3))

# model summary
print(data.rf2)


# plot rf
plot(data.rf2)


# accuracy went down
# let's check variable importance once again
# Variable Importance by Information gain
varImpPlot(data.rf2,  
           sort = T,
           n.var=5,
           main="Top 5 - Variable Importance")
# now, new user variable is the most important and source doesn't seem to matter at all.

# finally, let’s check partial dependence plots for the 4 vars:
# Partial dependence plot gives a graphical depiction of the marginal effect of a variable 
# on the class probability (classification) or response (regression).

op <- par(mfrow=c(2, 2))
partialPlot(data.rf2, as.data.frame(trainData2), country, 1)
partialPlot(data.rf2, as.data.frame(trainData2), age, 1)
partialPlot(data.rf2, as.data.frame(trainData2), new_user, 1)
partialPlot(data.rf2, as.data.frame(trainData2), source, 1)

# In partial dependence plots, we just care about the trend, not the actual y value. So this shows that:
# - Users with an old account are much better than new users
# - China is really bad, all other countries are similar with Germany being the best
# - The site works very well for young people and bad for less young people (>30 yrs old)
# - Source is irrelevant

