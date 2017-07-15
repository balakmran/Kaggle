# Load libraries
library(ggplot2)
library(gridExtra)
library(caret)

# Import Glass Dataset
Glass = read.csv("../input/glass.csv", sep = ",", header = T)

head(Glass)
summary(Glass)

# Outlier Analysis

# Boxplot for Feature Attributes
Na <- ggplot(Glass, aes(x = Type, y = Na, group = Type)) +  
  geom_boxplot(outlier.colour="red", outlier.shape=16, outlier.size=4) +
  geom_jitter(width = 0.1)

Mg <- ggplot(Glass, aes(x = Type, y = Mg, group = Type)) +
  geom_boxplot(outlier.colour="red", outlier.shape=16, outlier.size=4) +
  geom_jitter(width = 0.1)

grid.arrange(Na, Mg, ncol=2)

Al <- ggplot(Glass, aes(x = Type, y = Al, group = Type)) +
  geom_boxplot(outlier.colour="red", outlier.shape=16, outlier.size=4) + 
  geom_jitter(width = 0.1)

K  <- ggplot(Glass, aes(x = Type, y = K, group = Type)) +
  geom_boxplot(outlier.colour="red", outlier.shape=16, outlier.size=4) +
  geom_jitter(width = 0.1)

grid.arrange(Al, K, ncol=2)

Ca <- ggplot(Glass, aes(x = Type, y = Ca, group = Type)) +
  geom_boxplot(outlier.colour="red", outlier.shape=16, outlier.size=4) +
  geom_jitter(width = 0.1)

Ba <- ggplot(glass, aes(x = Type, y = Ba, group = Type)) +
  geom_boxplot(outlier.colour="red", outlier.shape=16, outlier.size=4) +
  geom_jitter(width = 0.1)

grid.arrange(Ca, Ba, ncol=2)

# Train Test Split
set.seed(1)
index <- createDataPartition(Glass$Type, p = .8, list = FALSE, times = 1)
gTrain <- Glass[index, ]
gTest <- Glass[-index, ]

# Boosting
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3)
perfMetric <- "Accuracy"

modelLookup(model='C5.0')

# C5.0 - Performace Tuning Grid
c50Grid <-  expand.grid(trials = (1:20)*5, 
                        model = c("tree", "rules"),
                        winnow = c(TRUE, FALSE)
                        )
set.seed(1)
mdl_c50 <- train(Type ~ ., data=gTrain, method="C5.0", metric=perfMetric,
                 trControl=ctrl, tuneGrid=c50Grid)

plot(mdl_c50)

# C5.0 - Evaluate Model on Testing Set
pred_c50 <- predict(mdl_c50, newdata = gTest, type = "raw")
tbl_c50 <- table(pred_c50, gTest$Type)
confusionMatrix(tbl_c50)

# Stochastic Gradient Boosting
modelLookup(model='gbm')
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)
set.seed(1)
mdl_gbm <- train(Type ~ ., data=gTrain, method="gbm", metric=perfMetric,
                 trControl=ctrl, verbose=FALSE, tuneGrid=gbmGrid)

trellis.par.set(caretTheme())
plot(mdl_gbm)
plot(varImp(object=mdl_gbm), main="GBM - Variable Importance")

# GBM - Evaluate Model on Testing Set
pred_gbm <- predict(mdl_gbm, newdata = gTest, type = "raw")
tbl_gbm <- table(pred_gbm, gTest$Type)
confusionMatrix(tbl_gbm)

# Boosting Results - C5.0 and GBM
boostingResults <- resamples(list(C50=mdl_c50, GBM=mdl_gbm))
summary(boostingResults)
bwplot(boostingResults)
densityplot(mdl_gbm)

# Bagged CART
set.seed(1)
mdl_treebag <- train(Type ~ ., data=gTrain, method="treebag",
                     metric=perfMetric, trControl=ctrl)
mdl_treebag

plot(varImp(object=mdl_treebag))

# TreeBag - Evaluate Model on Testing Set
pred_treebag <- predict(mdl_treebag, newdata = gTest, type = "raw")
tbl_treebag <- table(pred_treebag, gTest$Type)
confusionMatrix(tbl_treebag)

# Random Forest
rfGrid <- expand.grid(mtry=(1:5))
set.seed(1)
mdl_rf <- train(Type ~ ., data=gTrain, method="rf", metric=perfMetric,
                trControl=ctrl, tuneGrid=rfGrid)

plot(mdl_rf)
plot(varImp(object=mdl_rf))

# RF - Evaluate Model on Testing Set
pred_rf <- predict(mdl_rf, newdata = gTest, type = "raw")
tbl_rf <- table(pred_rf, gTest$Type)
confusionMatrix(tbl_rf)

# Bagging Results - Bagged CART and Random Forest
baggingResults <- resamples(list(BaggedCART=mdl_treebag, RandomForest=mdl_rf))
summary(baggingResults)
bwplot(baggingResults)

