# Adresse du dossier où vous travaillez
setwd("/home/charles/github/CREMIfiles/IF344_Analyse/apprentissageSuperviseClassification/Code")

# Packages utilisés dans la suite
library(rpart)
# Supprimer toutes les variables
rm(list=ls(all=TRUE))
# Supprimer tous les graphiques déjà présents
graphics.off()

# Lecture des données d’apprentissage
data_train <- read.table("../../Data/synth_train.txt", header=T, sep="\t");
plot(data_train$x1, data_train$x2, pch=data_train$y, col=data_train$y, main="Training set")
legend("topleft", legend=c("classe1", "classe2"), pch=1:2, col=1:2)

# Création de la sortie (à mettre sous le format facteur sinon
# un modèle de régression est créé)
data_train$y <- as.factor(data_train$y)

tree <- rpart(y~., data = data_train)

print(tree)
summary(tree)

plot(tree)
text(tree)

# Valeur de séparation de x1
split_x1 <- tree$splits[1,4]

data_test <- read.table("../../Data/synth_test.txt", header=T, sep="\t");
par(mfrow=c(1:2))
plot(data_train$x1, data_train$x2, pch=data_train$y, col=data_train$y, main="Training set")
legend("topleft", legend=c("classe1", "classe2"), pch=1:2, col=1:2)
abline(v=split_x1, lty=2, col=4)
plot(data_test$x1, data_test$x2, pch=data_test$y, col=data_test$y, main="Test set")
legend("topleft", legend=c("classe1", "classe2"), pch=1:2, col=1:2)
abline(v=split_x1, lty=2, col=4)

# Prediction sur les données test
data_test_x <- data.frame(x1=data_test$x1, x2=data_test$x2)
tree_predict_data_test <- predict(tree, newdata=data_test_x, type="class")

# Comparaison des valeurs prédites et des valeurs observées
table(tree_predict_data_test, data_test$y)

# Calcul du taux d’erreur
error_rate <- mean(tree_predict_data_test != data_test$y)
cat("error_rate using test data = ", error_rate)

# Arbre de longueur maximale
tree_max <- rpart(y~., data=data_train, minsplit=2,cp=0)
# Tracer l’arbre maximal
par(mfrow=c(1,1))
plot(tree_max)
text(tree_max,use.n = TRUE,all=TRUE)

# Affichage de l’arbre
plotcp(tree_max)

