# Adresse du dossier où vous travaillez
setwd("/home/charles/github/CREMIfiles/IF344_Analyse/apprentissageSuperviseClassification/Code")
# Packages utilisés dans la suite
library(randomForest)
# Supprimer toutes les variables
rm(list=ls(all=TRUE))
# Supprimer tous les graphiques déjà présents
graphics.off()

# Lecture des données d’apprentissage
data_train <- read.table("../../Data/synth_train.txt", header=T, sep="\t");
# Séparation des données et de la sortie !!
data_train_x <- data.frame(x1=data_train$x1, x2=data_train$x2)
# Création de la sortie (à mettre sous le format facteur sinon
# un modèle de régression est créé)
data_train_y <- as.factor(data_train$y)

# Forêts aléatoires
rf <- randomForest(x=data_train_x, y=data_train_y)

# Évolution de l’erreur en fonction du nombre d’arbres
# Ici ntree est fixé à la valeur par défaut = 500
plot(rf$err.rate[,1], type="l")

# Affichage des résultats
print(rf)

# Importance des variables
rf$importance
varImpPlot(rf)

# Lecture des données test
data_test <- read.table("../../Data/synth_test.txt", header=T, sep="\t");
# Séparation des données et de la sortie !!
data_test_x <- data.frame(x1=data_test$x1, x2=data_test$x2)

# Prediction sur les données test
rf_predit_data_test <- predict(rf, newdata=data_test_x)

# Comparaison des valeurs prédites et des valeurs observées
table(rf_predit_data_test, data_test$y)

# Calcul du taux d’erreur
error_rate <- mean(rf_predit_data_test != data_test$y)
cat("error_rate using test data = ", error_rate)

# Résultat de la méthode des forêts aléatoires
gridx1 <- seq(from=min(data_train$x1), to=max(data_train$x1), length.out=50)
gridx2 <- seq(from=min(data_train$x2), to=max(data_train$x2), length.out=50)
grid <- expand.grid(x1=gridx1, x2=gridx2)
data_grid_x <- data.frame(x1=grid[,1], x2=grid[,2])

rf_predit_data_grid <- predict(rf, newdata=data_grid_x)

plot(data_train$x1, data_train$x2, col=data_train$y, pch=16)
par(new=T)
plot(data_grid_x$x1, data_grid_x$x2, col=rf_predit_data_grid, pch=8, cex=0.5, ann=FALSE)
