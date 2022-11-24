# Adresse du dossier où vous travaillez
setwd("/home/charles/github/CREMIfiles/IF344_Analyse/apprentissageSuperviseClassification/Code")
# Packages utilisés dans la suite
library(MASS)
# Supprimer toutes les variables
rm(list=ls(all=TRUE))
# Supprimer tous les graphiques déjà présents
graphics.off()

# Lecture des données d’apprentissage
data_train <- read.table("../../Data/synth_train.txt", header=T, sep="\t");
data_train_y <- data_train$y
data_train$y <- as.factor(data_train$y)

# Analyse discriminante LINEAIRE
lin_disc_an <- lda(y~., data = data_train)

# Lecture des données test
data_test <- read.table("../../Data/synth_test.txt", header=T, sep="\t");
data_test_x <- data.frame(x1=data_test$x1, x2=data_test$x2)
# Prédiction sur les données test
test_LDA_predict <- predict(lin_disc_an, newdata=data_test_x, type="class")

# Comparaison des valeurs prédites et des valeurs observées
table(test_LDA_predict$class, data_test$y)
# Calcul du taux d’erreur
lda_error_rate <- mean(test_LDA_predict$class != data_test$y)
cat("error rate using test data (LDA) = ", lda_error_rate)

# Analyse discriminante QUADRATIQUE
quad_disc_an <- qda(y~., data = data_train)
# Prédiction sur les données test
test_QDA_predict <- predict(quad_disc_an, newdata=data_test_x, type="class")

# Comparaison des valeurs prédites et des valeurs observées
table(test_QDA_predict$class, data_test$y)
# Calcul du taux d’erreur
qda_error_rate <- mean(test_QDA_predict$class != data_test$y)
cat("error rate using test data (QDA) = ", qda_error_rate)

# Frontières LDA et QDA
gridx1 <- seq(from=min(data_train$x1), to=max(data_train$x1), length.out=40)
gridx2 <- seq(from=min(data_train$x2), to=max(data_train$x2), length.out=40)
grid <- expand.grid(x1 = gridx1, x2 = gridx2)
data_grid_x <- data.frame(x1=grid[,1], x2=grid[,2])

test_LDA_predict_grid <- predict(lin_disc_an, newdata=data_grid_x, type="class")
test_QDA_predict_grid <- predict(quad_disc_an, newdata=data_grid_x, type="class")

par(mfrow=c(1:2))
plot(data_train$x1, data_train$x2, col=data_train_y, pch=16)
par(new=T)
plot(data_grid_x$x1, data_grid_x$x2, col=test_LDA_predict_grid$class, pch=8, cex=0.5, ann=FALSE)

plot(data_train$x1, data_train$x2, col=data_train_y, pch=16)
par(new=T)
plot(data_grid_x$x1, data_grid_x$x2, col=test_QDA_predict_grid$class, pch=8, cex=0.5, ann=FALSE)


