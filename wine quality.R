## load packages that are required

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")

## set seed at the start to ensure consistency of output

set.seed(8934, sample.kind = "Rounding")

## download the red wine quality dataset

wine_quality <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                         sep = ";")

## have an initial look at the dataset

head(wine_quality)
dim(wine_quality)

## take a look at the dataset

# first up - take a look at the quality
wine_quality %>% ggplot(aes(x = quality)) +
  geom_histogram(binwidth = 0.5) +
  theme_classic()

# next - try a box plot to understand the distribution of the different variables
# that we have data for and may determine quality

wine_quality %>%
  gather(key, value) %>%
  ggplot(aes(y = value, x = key)) +
  geom_boxplot() +
  facet_wrap(~ key, scales = "free")

# define a quality wine as having a score of 7 or above

y <- if_else(wine_quality$quality >= 7, "quality", "sub-quality")
y <- factor(y)
data.frame(y) %>% 
  group_by(y) %>% 
  summarise(n()) %>% 
  knitr::kable()

## split into a test and training set with a 20% partition

# Validation set will be 20% of the total dataset

test_index <- createDataPartition(y = wine_quality$quality, times = 1, 
                                  p = 0.2, list = FALSE)
training_set <- wine_quality[-test_index,]
test_set <- wine_quality[test_index,]
y_training_set <- y[-test_index]
y_test_set <- y[test_index]

# check the proportion of the wines that are quality in both the test set and the training set

mean(y_training_set == "quality")
mean(y_test_set == "quality")

# lets now start with the simplest predictive model and see whether a simple correlation would
# give a high degree of accuracy

cor(wine_quality[1:11], wine_quality[12]) %>%
  knitr::kable()

# of these variables, it appears that alcohol has the highest correlation with quality, so let's build a predicitive
# based on this simple correlation

lm_fit <- mutate(training_set, y = as.numeric(quality >= 7)) %>% lm(y ~ alcohol, data = .)
p_hat <- predict(lm_fit, test_set)
y_hat <- ifelse(p_hat > 0.5, "quality", "sub-quality") %>% factor()
mean(y_hat == y_test_set)
wine_results <- data_frame(method = "regression on alcohol", 
                           accuracy = 
                             confusionMatrix(y_hat, y_test_set)$overall["Accuracy"],
                           sensitivity = 
                             confusionMatrix(y_hat, y_test_set)$byClass["Sensitivity"],
                           specificity = 
                             confusionMatrix(y_hat, y_test_set)$byClass["Specificity"],
                           balanced_acc = 
                             confusionMatrix(y_hat, y_test_set)$byClass["Balanced Accuracy"],
                           positive_pred_val = 
                             confusionMatrix(y_hat, y_test_set)$byClass["Pos Pred Value"],
                           negative_pred_val = 
                             confusionMatrix(y_hat, y_test_set)$byClass["Neg Pred Value"]) 

# Remove the quality value from the data table to ease the application of machine learning approaches

training_set <- training_set[,-12]
test_set <- test_set[,-12]

# start with a simple linear regression model to see what results that will give

train_glm <- train(training_set, y_training_set,
                   method = "glm")
glm_preds <- predict(train_glm, test_set)
mean(glm_preds == y_test_set)
wine_results <- bind_rows(wine_results,
                          data_frame(method="general linear model",
                                     accuracy = 
                                       confusionMatrix(glm_preds, 
                                                       y_test_set)$overall["Accuracy"],
                                     sensitivity = 
                                       confusionMatrix(glm_preds, 
                                                       y_test_set)$byClass["Sensitivity"],
                                     specificity = 
                                       confusionMatrix(glm_preds, 
                                                       y_test_set)$byClass["Specificity"],
                                     balanced_acc = 
                                       confusionMatrix(glm_preds, 
                                                       y_test_set)$byClass["Balanced Accuracy"],
                                     positive_pred_val = 
                                       confusionMatrix(glm_preds, 
                                                       y_test_set)$byClass["Pos Pred Value"],
                                     negative_pred_val = 
                                       confusionMatrix(glm_preds, 
                                                       y_test_set)$byClass["Neg Pred Value"]))

# next try the LDA model

train_lda <- train(training_set, y_training_set,
                   method = "lda")
lda_preds <- predict(train_lda, test_set)
mean(lda_preds == y_test_set)
wine_results <- bind_rows(wine_results,
                          data_frame(method="LDA model",
                                     accuracy = 
                                       confusionMatrix(lda_preds, 
                                                       y_test_set)$overall["Accuracy"],
                                     sensitivity = 
                                       confusionMatrix(lda_preds, 
                                                       y_test_set)$byClass["Sensitivity"],
                                     specificity = 
                                       confusionMatrix(lda_preds, 
                                                       y_test_set)$byClass["Specificity"],
                                     balanced_acc = 
                                       confusionMatrix(lda_preds, 
                                                       y_test_set)$byClass["Balanced Accuracy"],
                                     positive_pred_val = 
                                       confusionMatrix(lda_preds, 
                                                       y_test_set)$byClass["Pos Pred Value"],
                                     negative_pred_val = 
                                       confusionMatrix(lda_preds, 
                                                       y_test_set)$byClass["Neg Pred Value"]))


# and the QDA model

train_qda <- train(training_set, y_training_set,
                   method = "qda")
qda_preds <- predict(train_qda, test_set)
mean(qda_preds == y_test_set)
wine_results <- bind_rows(wine_results,
                          data_frame(method="QDA model",
                                     accuracy = 
                                       confusionMatrix(qda_preds, 
                                                       y_test_set)$overall["Accuracy"],
                                     sensitivity = 
                                       confusionMatrix(qda_preds, 
                                                       y_test_set)$byClass["Sensitivity"],
                                     specificity = 
                                       confusionMatrix(qda_preds, 
                                                       y_test_set)$byClass["Specificity"],
                                     balanced_acc = 
                                       confusionMatrix(qda_preds, 
                                                       y_test_set)$byClass["Balanced Accuracy"],
                                     positive_pred_val = 
                                       confusionMatrix(qda_preds, 
                                                       y_test_set)$byClass["Pos Pred Value"],
                                     negative_pred_val = 
                                       confusionMatrix(qda_preds, 
                                                       y_test_set)$byClass["Neg Pred Value"]))

# and a Loess model

train_loess <- train(training_set, y_training_set,
                   method = "gamLoess")
loess_preds <- predict(train_loess, test_set)
mean(loess_preds == y_test_set)
wine_results <- bind_rows(wine_results,
                          data_frame(method="Loess",
                                     accuracy = 
                                       confusionMatrix(loess_preds, 
                                                       y_test_set)$overall["Accuracy"],
                                     sensitivity = 
                                       confusionMatrix(loess_preds, 
                                                       y_test_set)$byClass["Sensitivity"],
                                     specificity = 
                                       confusionMatrix(loess_preds, 
                                                       y_test_set)$byClass["Specificity"],
                                     balanced_acc = 
                                       confusionMatrix(loess_preds, 
                                                       y_test_set)$byClass["Balanced Accuracy"],
                                     positive_pred_val = 
                                       confusionMatrix(loess_preds, 
                                                       y_test_set)$byClass["Pos Pred Value"],
                                     negative_pred_val = 
                                       confusionMatrix(loess_preds, 
                                                       y_test_set)$byClass["Neg Pred Value"]))

# and a random forest model

train_rf <- train(training_set, y_training_set,
                     method = "rf")
rf_preds <- predict(train_rf, test_set)
mean(rf_preds == y_test_set)
wine_results <- bind_rows(wine_results,
                          data_frame(method="Random Forest",
                                     accuracy = 
                                       confusionMatrix(rf_preds, 
                                                       y_test_set)$overall["Accuracy"],
                                     sensitivity = 
                                       confusionMatrix(rf_preds, 
                                                       y_test_set)$byClass["Sensitivity"],
                                     specificity = 
                                       confusionMatrix(rf_preds, 
                                                       y_test_set)$byClass["Specificity"],
                                     balanced_acc = 
                                       confusionMatrix(rf_preds, 
                                                       y_test_set)$byClass["Balanced Accuracy"],
                                     positive_pred_val = 
                                       confusionMatrix(rf_preds, 
                                                       y_test_set)$byClass["Pos Pred Value"],
                                     negative_pred_val = 
                                       confusionMatrix(rf_preds, 
                                                       y_test_set)$byClass["Neg Pred Value"]))


# and a knn model

train_knn <- train(training_set, y_training_set,
                  method = "knn")
knn_preds <- predict(train_knn, test_set)
mean(knn_preds == y_test_set)
wine_results <- bind_rows(wine_results,
                          data_frame(method="k nearest neighbour",
                                     accuracy = 
                                       confusionMatrix(knn_preds, 
                                                       y_test_set)$overall["Accuracy"],
                                     sensitivity = 
                                       confusionMatrix(knn_preds, 
                                                       y_test_set)$byClass["Sensitivity"],
                                     specificity = 
                                       confusionMatrix(knn_preds, 
                                                       y_test_set)$byClass["Specificity"],
                                     balanced_acc = 
                                       confusionMatrix(knn_preds, 
                                                       y_test_set)$byClass["Balanced Accuracy"],
                                     positive_pred_val = 
                                       confusionMatrix(knn_preds, 
                                                       y_test_set)$byClass["Pos Pred Value"],
                                     negative_pred_val = 
                                       confusionMatrix(knn_preds, 
                                                       y_test_set)$byClass["Neg Pred Value"]))

# and an adaboost model

train_adaboost <- train(training_set, y_training_set,
                   method = "adaboost")
adaboost_preds <- predict(train_adaboost, test_set)
mean(adaboost_preds == y_test_set)
wine_results <- bind_rows(wine_results,
                          data_frame(method="adaboost",
                                     accuracy = 
                                       confusionMatrix(adaboost_preds, 
                                                       y_test_set)$overall["Accuracy"],
                                     sensitivity = 
                                       confusionMatrix(adaboost_preds, 
                                                       y_test_set)$byClass["Sensitivity"],
                                     specificity = 
                                       confusionMatrix(adaboost_preds, 
                                                       y_test_set)$byClass["Specificity"],
                                     balanced_acc = 
                                       confusionMatrix(adaboost_preds, 
                                                       y_test_set)$byClass["Balanced Accuracy"],
                                     positive_pred_val = 
                                       confusionMatrix(adaboost_preds, 
                                                       y_test_set)$byClass["Pos Pred Value"],
                                     negative_pred_val = 
                                       confusionMatrix(adaboost_preds, 
                                                       y_test_set)$byClass["Neg Pred Value"]))

# and a neural network model

train_nnet <- train(training_set, y_training_set,
                        method = "nnet")
nnet_preds <- predict(train_nnet, test_set)
mean(nnet_preds == y_test_set)
wine_results <- bind_rows(wine_results,
                          data_frame(method="neural network",
                                     accuracy = 
                                       confusionMatrix(nnet_preds, 
                                                       y_test_set)$overall["Accuracy"],
                                     sensitivity = 
                                       confusionMatrix(nnet_preds, 
                                                       y_test_set)$byClass["Sensitivity"],
                                     specificity = 
                                       confusionMatrix(nnet_preds, 
                                                       y_test_set)$byClass["Specificity"],
                                     balanced_acc = 
                                       confusionMatrix(nnet_preds, 
                                                       y_test_set)$byClass["Balanced Accuracy"],
                                     positive_pred_val = 
                                       confusionMatrix(nnet_preds, 
                                                       y_test_set)$byClass["Pos Pred Value"],
                                     negative_pred_val = 
                                       confusionMatrix(nnet_preds, 
                                                       y_test_set)$byClass["Neg Pred Value"]))

# finally include an ensemble model, which combines takes an average across all the models

ensemble <- cbind(glm = glm_preds == "quality", lda = lda_preds == "quality", 
                  qda = qda_preds == "quality", 
                  loess = loess_preds == "quality", rf = rf_preds == "quality", 
                  knn = knn_preds == "quality", adaboost = adaboost_preds == "quality", 
                  nnet = nnet_preds == "quality")

ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, "quality", "sub-quality") %>% factor()
mean(ensemble_preds == y_test_set)
wine_results <- bind_rows(wine_results,
                          data_frame(method="ensemble",
                                     accuracy = 
                                       confusionMatrix(ensemble_preds, 
                                                       y_test_set)$overall["Accuracy"],
                                     sensitivity = 
                                       confusionMatrix(ensemble_preds, 
                                                                   y_test_set)$byClass["Sensitivity"],
                                     specificity = 
                                       confusionMatrix(ensemble_preds, 
                                                       y_test_set)$byClass["Specificity"],
                                     balanced_acc = 
                                       confusionMatrix(ensemble_preds, 
                                                       y_test_set)$byClass["Balanced Accuracy"],
                                     positive_pred_val = 
                                       confusionMatrix(ensemble_preds, 
                                                       y_test_set)$byClass["Pos Pred Value"],
                                     negative_pred_val = 
                                       confusionMatrix(ensemble_preds, 
                                                       y_test_set)$byClass["Neg Pred Value"]))


# print the results

wine_results %>% 
  knitr::kable()


