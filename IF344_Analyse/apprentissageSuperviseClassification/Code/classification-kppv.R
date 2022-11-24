# Adresse du dossier où vous travaillez
setwd("/home/charles/github/CREMIfiles/IF344_Analyse/apprentissageSuperviseClassification/Code")

# Packages utilisés dans la suite
library(class)
library(caret)
library(ROCR)

# Supprimer toutes les variables
rm(list=ls(all=TRUE))

# Supprimer tous les graphiques déjà présents
graphics.off()

# Lecture des données d’apprentissage
data_train <- read.table("../../Data/synth_train.txt", header=T, sep="\t");
print(data_train)

# Séparation des données et de la sortie
data_train_x <- data.frame(x1=data_train$x1, x2=data_train$x2)

# Lecture des données test
data_test <- read.table("../../Data/synth_test.txt", header=T, sep="\t");
print(data_test)
# Séparation des données et de la sortie
data_test_x <- data.frame(x1=data_test$x1, x2=data_test$x2)

# Graphique des données (colorées par la sortie y)
plot(data_train$x1, data_train$x2, col=data_train$y, pch=16)

# k-plus proches voisins (knn)
# nombre de voisins (par ex proche de la racine carré du nombre d’obs)
num_of_neigh <- 1
data_train_predict <- knn(train=data_train_x, test=data_train_x, cl=data_train$y, k=num_of_neigh)

# Affichage des résultats (étoile)
par(new=T)
plot(data_train$x1, data_train$x2, col=data_train_predict, pch=8)

# Calcul du taux d’erreur
error_rate <- mean(data_train_predict != data_train$y)
cat("error_rate using train data = ", error_rate)

# test
data_test_predict <- knn(train=data_train_x, test=data_test_x, cl=data_train$y, k=num_of_neigh)

# Affichage des données (cercle)
plot(data_train$x1, data_train$x2, col=data_train$y, pch=16)
par(new=T)

# Affichage des résultats (étoile)
plot(data_test$x1, data_test$x2, col=data_test_predict, pch=8)

# Affichage des vraies valeurs (triangle)
par(new=T)
plot(data_test$x1, data_test$x2, col=data_test$y, pch=2)

errorRateAndConfusionMatrix <- function (X, y) {
  # Calcul du taux d’erreur
  error_rate <- mean(data_test_predict != data_test$y)
  cat("error_rate using test data = ", error_rate)

  # Matrice de confusion
  confmat = table(X, y)
  print("Confusion Matrix")
  print(confmat)
  # vrais positifs + vrais negatifs + faux positifs + faux négatifs
  TP = confmat[1,1]; TN = confmat[2,2]; FP = confmat[1,2]; FN = confmat[2,1];

  # Sensibilité (sensitivity ; TPR = true positive rate)
  TPR = TP/(TP+FN)
  cat("Sensibilité (TPR)", TPR,"\n")
  # Spécificité (specificity ; TNR = true negative rate)
  TNR = TN/(TN+FP)
  cat("Spécificité (TNR)", TNR,"\n")
  # Précision (precision ; positive predictive value)
  PPV = TP/(TP+FP)
  cat("Précision (PPV)", PPV,"\n")
  # se compare à la prévalence (prevalence)
  cat("Prev =", length(y[y==1]) / length(y), "\n")

  cat("F-score = ",2 * TPR * PPV / (TPR+PPV), "\n")
}
errorRateAndConfusionMatrix(data_test_predict, data_test$y)

# Explication visuelle de l’importance de la valeur de k (nb de voisins)
# Construction de la grille
gridx1 <- seq(from=min(data_train$x1), to=max(data_train$x1), length.out=50)
gridx2 <- seq(from=min(data_train$x2), to=max(data_train$x2), length.out=50)
grid <- expand.grid(x1 = gridx1, x2 = gridx2)
data_grid_x <- data.frame(x1=grid[,1], x2=grid[,2])

# k plus proches voisins avec application sur les données de la grille
num_of_neigh_grid <- c(1, 5, 10, 15, 20, 30)
par(mfrow=c(2, length(num_of_neigh_grid)/2))
for (i in 1:length(num_of_neigh_grid)) {
  num_of_n <- num_of_neigh_grid[i]
  data_g_pr <- knn(train=data_train_x, test=data_grid_x, cl=data_train$y, k=num_of_n)
  plot(data_train$x1,data_train$x2, col=data_train$y, pch=16)
  title(paste("num of neighbours = ", toString(num_of_n)))
  par(new=T)
  plot(data_grid_x$x1, data_grid_x$x2, col=data_g_pr, pch=8, cex=0.5, ann=FALSE)
}

for (n in c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)) {
  num_of_neigh <- n
  data_test_predict <- knn(train=data_train_x, test=data_test_x, cl=data_train$y, k=num_of_neigh)

  cat("\ntests knn with k=", num_of_neigh, "\n")
  errorRateAndConfusionMatrix(data_test_predict, data_test$y)
}

# k plus proches voisins avec les probas
data_test_predict_with_proba <- knn(train=data_train_x, test=data_test_x, cl=data_train$y, k=num_of_neigh, prob=TRUE)

# Calcul du score
score <- attr(data_test_predict_with_proba, "prob")
score <- ifelse(data_test_predict_with_proba == "1", 1-score, score)

# Courbe ROC
pred_knn <- prediction(score, data_test$y)
perf <- performance(pred_knn, "tpr", "fpr")
par(mfrow=c(1, 1))
plot(perf, colorize=TRUE)
par(new=T)
plot(c(0, 1), c(0, 1), type="l", ann=FALSE)

# Aire sous la courbe
AUC <- performance(pred_knn, "auc")@y.values[[1]]
cat("AUC = ", AUC)

# Choix du seuil
result <- NULL
threshold <- seq(0,1, len=11)
for (s in threshold) {
  test <- as.integer(score>=s)+1
  result <- c(result, 1-mean(test != data_test$y))
}
plot(threshold, result, type="l")
cat("Meilleur seuil ", threshold[which.max(result)])

