# Loading the Iris data set

library(datasets)
data(iris)

iris <- datasets::iris

#View the data set
View(iris)

#Display summary statistics

head(iris, 4)
tail(iris, 4)

summary(iris)
summary(iris$Sepal.Length)

#Check to see if there is are any missing data

sum(is.na(iris))

#skimr() - expands on summary() by providing larger set of statistics
install.packages("skimr")

library(skimr)
library(dplyr)

skimr(iris) # Perform skim to display summary statistics

# Group data by Species then perform skim
iris %>%
  dplyr::group_by(Species) %>%
  skim()

# Quick Visualisations
# R base plot()

#Panel plots
plot(iris)
plot(iris, col = "red")

#Scatter Plot
plot(iris$Sepal.Width, iris$Sepal.Length)

plot(iris$Sepal.Width, iris$Sepal.Length, col = "red")  # Makes red circles

plot(iris$Sepal.Width, iris$Sepal.Length, col = "red", xlab = 'Sepal Width', ylab = 'Sepal Length')

# Histogram
hist(iris$Sepal.Width)
hist(iris$Sepal.Width, col = "red")

install.packages('caret')
library(caret)

# Feature Plots
featurePlot(x = iris[,1:4],
            y = iris$Species,
            plot = "box",
            strip=strip.custom(par.strip.text = list(cex = .7)),
            scales = list(x = list(relation = "free"),
                          y = list(relation = "free")))

# To achieve a reproducible model; set the random seed number
set.seed(100)

# Perform stratified random split of the data set
TrainingIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
TrainingSet <- iris[TrainingIndex,]
TestSet <- iris[-TrainingIndex,]

# Compare scatter plot of the 80 and 20 data subsets

plot(TrainingSet$Sepal.Width, TrainingSet$Sepal.Length, col = 'cyan')
plot(TestSet$Sepal.Width, TestSet$Sepal.Length, col = 'pink')

# SVM model (polynomial kernel)

# Build training model
Model <- train(Species ~., data = TrainingSet,
               method = "svmPoly",
               na.action = na.omit,
               preProcess = c("scale", "center"),
               trControl = trainControl(method ="cv", number = 10),
               tuneGrid = data.frame(degree = 1, scale = 1, C = 1))

# Build CV model
Model.cv <- train(Species ~ ., data = TrainingSet,
                  method = "svmPoly",
                  na.action = na.omit,
                  preProcess = c("scale", "center"),
                  trControl = trainControl(method="cv", number=10),
                  tuneGrid = data.frame(degree=1, scale=1, C=1))

# Apply model for prediction
Model.training <- predict(Model, TrainingSet)
Model.testing <- predict(Model, TestSet)
Model.cv <- predict(Model.cv, TrainingSet)

# Model performance (Displays confusion matrix and statistics)
Model.training.confusion <- confusionMatrix(Model.training,TrainingSet$Species)
Model.testing.confusion <- confusionMatrix(Model.testing, TestSet$Species)
Model.cv.confusion <- confusionMatrix(Model.cv, TrainingSet$Species)

print(Model.training.confusion)
print(Model.testing.confusion)
print(Model.cv.confusion)

# Feature importance
Importance <- varImp(Model)
plot(Importance)
plot(Importance, col = "red")


