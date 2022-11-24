# Adresse du dossier où vous travaillez
setwd("/home/charles/github/CREMIfiles/IF344_Analyse/apprentissageSuperviseRegression/Code")

# Packages utilisés dans la suite
library(MASS)
require(pls)
require(splines)

# Supprimer toutes les variables
rm(list=ls(all=TRUE))

# Utilisation de données sur data
# Affichage des informations
?Boston # help on Boston dataset from the lib MASS

# Affichage des données
print(Boston)

# Transformation des données
data <- Boston
data <- data.frame(
  y=Boston$medv,
  x1=Boston$lstat,
  x2=Boston$age,
  x3=Boston$crim,
  x4=Boston$zn,
  x5=Boston$indus,
  x6=Boston$chas,
  x7=Boston$nox,
  x8=Boston$rm,
  x9=Boston$dis,
  x10=Boston$rad,
  x11=Boston$tax,
  x12=Boston$ptratio,
  x13=Boston$black
)

# Paramètres
n <- length(data$y) # nb rows
alpha <- 0.05

## Mise en place de la régression linéaire [SIMPLE]
# Peut on utiliser x1 = pourcentage de la population pauvre
# pour prédire y = valeur médiane des maisons en milliers de dollars.
simpleLinearReg <- lm(y~x1, data=data) # fitting Linear Models

# Affichage du résultat de la régression linéaire
# épaisseur de la ligne = 2 ; couleur de la ligne = rouge
plot(y~x1, data=data)
abline(simpleLinearReg, lwd=2, col="red") # lwd : Line WiDth/thickness

# Affichage des résidus en fonction de la prédiction
plot(simpleLinearReg$fitted.values, simpleLinearReg$residuals)
abline(0, 0)

# Affichage des valeurs prédites en fonction des valeurs observées
plot(simpleLinearReg$fitted.values, data$y)
abline(0, 1)

# Affichage du résultat : calculer le risque à 5 % avec :
# la t-value / la p-value/ la statistique de Fisher
summary(simpleLinearReg)
# Risque à 5 % (pour la t-value / la statistique de Fisher)
qt(1-alpha/2, n-2) # t-value
qf(1-alpha/2, 1, n-2) # alpha Fisher

# Intervalle de confiance des paramètres estimés
# Risque à 5 % avec l’intervalle de confiance
confint(simpleLinearReg)
# on fait l'interval de confiance à 95% pour avoir 5% d'erreur
confint(simpleLinearReg, level=1-alpha)

# Adéquation au modèle avec le R^2
summary(simpleLinearReg)

# Prédiction d’une valeur ultérieure (valeur de x1 testée = 10)
# Intervalle de confiance pour la prédiction de y pour une valeur donnée de x1
predict(simpleLinearReg, data.frame(x1=10), interval="confidence")
# Intervalle de prédiction pour la prédiction de y pour une valeur donnée de x1
predict(simpleLinearReg, data.frame(x1=10), interval="prediction")

# Affichage de l’intervalle de confiance et de prédiction
seqx1 <- seq(min(data$x1), max(data$x1), length=50)
intpred <- predict(simpleLinearReg, data.frame(x1=seqx1), interval="prediction")[,c("lwr", "upr")]
intconf <- predict(simpleLinearReg, data.frame(x1=seqx1), interval="confidence")[,c("lwr", "upr")]
plot(data$y~data$x1, xlab="x1", ylab="y")
abline(simpleLinearReg, lwd=2, col="red")
# cbind(a, b) = column bund, combine a & b ; lty = Line sTYle (1: line, 2: dashed, 3: dotted...)
matlines(seqx1, cbind(intconf, intpred), lty=c(2, 2, 3, 3), col=c("green3", "green3", "blue", "blue"), lwd=c(2, 2))
legend("topright", lty=c(1, 2, 3), lwd=c(2, 2, 2), c("linearReg", "conf", "pred"), col=c("red", "green3", "blue"))

# Test de normalité des résidus
shapiro.test(resid(simpleLinearReg))

# Validation du modèle par validation croisée
MSE <- 0
for (i in 1:n) {
  dataToPredict <- data$y[i]
  dataTemp <- data[-c(i),]
  reg <- lm(y~x1, data=dataTemp)
  predictedValue <- predict(reg, data.frame(x1=data$x1[i]), interval="prediction")
  MSE <- MSE+(dataToPredict-predictedValue[1])^2
}
MSE <- MSE/n
cat("Valeur du résidu avec la validation croisée", MSE)

## Et le non linéaire ?
## Cas polynomial (simple = une variable)
degpoly <- 2
simplePolyReg <- lm(y~poly(x1, degpoly), data=data)

# Risque à 5 % (pour la t-value / la statistique de Fisher)
qt(1-alpha/2, n-2)
qf(1-alpha/2, 1, n-2)

# Intervalle de confiance des paramètres estimés
# Risque à 5 % avec l’intervalle de confiance
confint(simplePolyReg)

# Adéquation au modèle avec le R^2
summary(simplePolyReg)

# Affichage de l’intervalle de confiance et de prédiction
seqx1 <- seq(min(data$x1), max(data$x1), length=50)
intpred <- predict(simplePolyReg, data.frame(x1=seqx1), interval="prediction")[,c("lwr", "upr")]
intconf <- predict(simplePolyReg, data.frame(x1=seqx1), interval="confidence")[,c("lwr", "upr")]
plot(data$y~data$x1, xlab="x1", ylab="y")
pred <- predict(simplePolyReg, data.frame(x1=sort(data$x1)))
lines(sort(data$x1), pred, lwd=2, col="red")
matlines(seqx1, cbind(intconf, intpred), lty=c(2, 2, 3, 3), col=c("green3", "green3", "blue", "blue"), lwd=c(2, 2))
legend("topright", lty=c(1, 2, 3), lwd=c(2, 2, 2), c("linearReg", "conf", "pred"), col=c("red", "green3", "blue"))

# Test de normalité des résidus
shapiro.test(resid(simplePolyReg))

# Validation du modèle par validation croisée
MSE <- 0
for (i in 1:n) {
  dataToPredict <- data$y[i]
  dataTemp <- data[-c(i),]
  reg <- lm(y~poly(x1, degpoly), data=dataTemp)
  predictedValue <- predict(reg, data.frame(x1=data$x1[i]), interval="prediction")
  MSE <- MSE+(dataToPredict-predictedValue[1])^2
}
MSE <- MSE/n
# Valeur du résidu avec la validation croisée
print(MSE)

## Et le non linéaire ?
## Cas spline (simple = une variable)
degoffreedom <- 4
simpleSplineReg <- lm(y~ns(x1, degoffreedom), data=data)

# Risque à 5 % (pour la t-value / la statistique de Fisher)
qt(1-alpha/2, n-2)
qf(1-alpha/2, 1, n-2)

# Intervalle de confiance des paramètres estimés
# Risque à 5 % avec l’intervalle de confiance
confint(simpleSplineReg)

# Adéquation au modèle avec le R^2
summary(simpleSplineReg)

# Affichage de l’intervalle de confiance et de prédiction
seqx1 <- seq(min(data$x1), max(data$x1), length=50)
intpred <- predict(simpleSplineReg, data.frame(x1=seqx1), interval="prediction")[,c("lwr", "upr")]
intconf <- predict(simpleSplineReg, data.frame(x1=seqx1), interval="confidence")[,c("lwr", "upr")]
plot(data$y~data$x1, xlab="x1", ylab="y")
pred <- predict(simpleSplineReg, data.frame(x1=sort(data$x1)))
lines(sort(data$x1), pred, lwd=2, col="red")
matlines(seqx1, cbind(intconf, intpred), lty=c(2, 2, 3, 3), col=c("green3", "green3", "blue", "blue"), lwd=c(2, 2))
legend("topright", lty=c(1, 2, 3), lwd=c(2, 2, 2), c("linearReg", "conf", "pred"), col=c("red", "green3", "blue"))

# Test de normalité des résidus
shapiro.test(resid(simpleSplineReg))

# Validation du modèle par validation croisée
MSE <- 0
for (i in 1:n) {
  datatopredict <- data$y[i]
  datatemp <- data[-c(i),]
  reg <- lm(y~ns(x1,degoffreedom), data=datatemp)
  predictedvalue <- predict(reg,data.frame(x1=data$x1[i]), interval="prediction")
  MSE <- MSE+(datatopredict-predictedvalue[1])^2
}
MSE <- MSE/n
# Valeur du résidu avec la validation croisée
print(MSE)

## Cas smoothing spline (simple = une variable)
degoffreedom <- 6
simpleSmoothSplineReg <- smooth.spline(data$x1, data$y, df=degoffreedom)

# Affichage de l’intervalle de confiance et de prédiction
plot(data$y~data$x1, xlab="x1", ylab="y")
pred <- predict(simpleSmoothSplineReg, sort(data$x1))
lines(pred, lwd=2, col="red")

# Test de normalité des résidus
shapiro.test(resid(simpleSmoothSplineReg))

# Validation du modèle par validation croisée
MSE <- 0
for (i in 1:n)
{
  datatopredict <- data$y[i]
  datatemp <- data[-c(i),]
  reg <- smooth.spline(datatemp$x1,datatemp$y, df=degoffreedom)
  predictedvalue <- predict(reg, data$x1[i])
  MSE <- MSE+(datatopredict-predictedvalue$y)^2
}
MSE <- MSE/n
# Valeur du résidu avec la validation croisée
print(MSE)

