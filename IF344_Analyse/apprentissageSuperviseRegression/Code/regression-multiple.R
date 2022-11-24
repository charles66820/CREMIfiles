# Adresse du dossier où vous travaillez
setwd("/home/charles/github/CREMIfiles/IF344_Analyse/apprentissageSuperviseRegression/Code")

# Packages utilisés dans la suite
library(MASS)
require(pls)

# Supprimer toutes les variables
rm(list=ls(all=TRUE))

# Utilisation de données sur data
# Affichage des informations
?Boston

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
n <- length(data$y)
alpha <- 0.05

## Mise en place de la régression linéaire [SIMPLE]
simpleLinearReg <- lm(y~x1, data=data)

# all tests
numOfVariables <- 1
qt(1-alpha/2, n-numOfVariables-1)
qf(1-alpha/2, numOfVariables, n-numOfVariables-1)

summary(simpleLinearReg)
confint(simpleLinearReg)
shapiro.test(resid(simpleLinearReg))


## Mise en place de la régression linéaire [MULTIPLE]
linearReg <- lm(y~x1+x2, data=data)

# Affichage du résultat
summary(linearReg)

numOfVariables <- 2
qt(1-alpha/2, n-numOfVariables-1)
qf(1-alpha/2, numOfVariables, n-numOfVariables-1)

# Intervalle de confiance
confint(linearReg)

# méthode anova
numOfVariablesToTest = 1
qf(1-alpha/2, numOfVariablesToTest, n-numOfVariables-1)
anova(simpleLinearReg,linearReg)
simpleLinearRegx2 <- lm(y~x2, data=data)
anova(simpleLinearRegx2,linearReg)

# Prédiction
predict(linearReg,data.frame(x1=10, x2=72), interval="confidence")
predict(linearReg,data.frame(x1=10, x2=72), interval="prediction")

# Affichage des résidus en fonction de la prédiction
plot(linearReg$fitted.values, linearReg$residuals)
abline(0,0)

# Test de normalité des résidus
shapiro.test(resid(linearReg))

## Selection de variables (en utilisant TOUTES les variables)
fullLinearReg <- lm(y~., data=data)
back <- step(fullLinearReg, direction="backward", trace = 1)
formula(back)
null <- lm(y ~ 1, data=data)
forw <- step(null, scope=list(lower=null, upper=fullLinearReg), direction="forward", trace = 1)
formula(forw)

# Utilisation de l’ACP pour réduire la dimension du problème
# Test sur validation croisée
# Combien de composantes pour avoir 80 % de variance expliquée ?
redDim = pcr(y~., data=data, scale=TRUE, validation="CV")
summary(redDim)

# all tests fullLinearReg
numOfVariables <- 13
qt(1-alpha/2, n-numOfVariables-1)
qf(1-alpha/2, numOfVariables, n-numOfVariables-1)

summary(fullLinearReg)
confint(fullLinearReg)
shapiro.test(resid(fullLinearReg))

# all tests selected11LinearReg
numOfVariables <- 11
qt(1-alpha/2, n-numOfVariables-1)
qf(1-alpha/2, numOfVariables, n-numOfVariables-1)

selected11LinearReg <- lm(y ~ x1 + x8 + x12 + x9 + x7 + x6 + x13 + x4 + x3 + x10 + x11, data=data)

summary(selected11LinearReg)
confint(selected11LinearReg)
shapiro.test(resid(selected11LinearReg))
