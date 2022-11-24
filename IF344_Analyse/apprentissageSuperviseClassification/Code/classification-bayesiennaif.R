# Adresse du dossier où vous travaillez
setwd("/home/charles/github/CREMIfiles/IF344_Analyse/apprentissageSuperviseClassification/Code")
# Packages utilisés dans la suite
library(e1071)
# Supprimer toutes les variables
rm(list=ls(all=TRUE))
# Supprimer tous les graphiques déjà présents
graphics.off()

# Lecture des données d’apprentissage
data_train <- read.table("../../Data/synth_train.txt", header=T, sep="\t");
data_train_y <- data_train$y
data_train$y <- as.factor(data_train$y)

# Bayesien Naif
naiv_bayes <- naiveBayes(y~., data = data_train)

# Lecture des données test
data_test <- read.table("../../Data/synth_test.txt", header=T, sep="\t");
data_test_x <- data.frame(x1=data_test$x1, x2=data_test$x2)
# Prédiction sur les données test
naiv_bayes_predict <- predict(naiv_bayes, newdata=data_test_x, type="class")

# Comparaison des valeurs prédites et des valeurs observées
table(naiv_bayes_predict, data_test$y)
# Calcul du taux d’erreur
naiv_bayes_error_rate <- mean(naiv_bayes_predict != data_test$y)
cat("error rate using test data (Naive bayes) = ", naiv_bayes_error_rate)

gridx1 <- seq(from=min(data_train$x1), to=max(data_train$x1), length.out=50)
gridx2 <- seq(from=min(data_train$x2), to=max(data_train$x2), length.out=50)
grid <- expand.grid(x1 = gridx1, x2 = gridx2)
data_grid_x <- data.frame(x1=grid[,1], x2=grid[,2])
naiv_bayes_predict_grid <- predict(naiv_bayes, newdata=data_grid_x, type="class")
plot(data_train$x1, data_train$x2, col=data_train_y, pch=16)
par(new=T)
plot(data_grid_x$x1, data_grid_x$x2, col=naiv_bayes_predict_grid, pch=8, cex=0.5, ann=FALSE)
