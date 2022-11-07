# Adresse du dossier où vous travaillez
setwd("/home/charles/github/CREMIfiles/IF344_Analyse/Code")

# Packages utilisés dans la suite
library("FactoMineR")

# Données sur les fromages
X <- read.table("../../Data/fromage.txt", sep="", header=TRUE, row.names=1)
print(X)

# Calcul de la moyenne et de l’écart type des variables
mean <- apply(X, 2, mean)
std <- apply(X, 2, sd) # standard deviation
stat <- rbind(mean, std)
# Affichage
print(stat, digits=4)

# Création des données centrées ...
Xnorm <- sweep(X, 2, mean, "-")
# ... et réduites
Xnorm <- sweep(Xnorm, 2, std, "/")
# Affichage des données centrées - réduites
print(Xnorm, digits=4)

# Nombre de clusters souhaité
numcluster <- 5

## KMEANS
# Algorithme des kmeans (avec affichage)
km <- kmeans(X, numcluster, nstart=50)
print(km)
# Algorithme des kmeans sur données centrées-réduites (avec affichage)
kmnorm <- kmeans(Xnorm, numcluster, nstart=50)
print(kmnorm)

# Concatenation des données avec leur résultat de cluster
cluster <- as.factor(km$cluster)
clusternorm <- as.factor(kmnorm$cluster)

XplusCluster <- data.frame(X, cluster=cluster)
XnormplusCluster <- data.frame(Xnorm, cluster=clusternorm)

colclust <- length(X) + 1

print(XplusCluster)
print(XnormplusCluster)

# ACP sur les données brutes
rPCA <- PCA(XplusCluster, scale.unit=FALSE, graph=FALSE, quali.sup=colclust)

# Nuage des individus et des variables dans le premier plan factoriel
par(mfrow=c(1, 2))
plot.PCA(rPCA, axes=c(1, 2), choix="ind", habillage=colclust, invisible="quali")
plot.PCA(rPCA, axes=c(1, 2), choix="var")

# Nuage des individus et des variables dans le deuxième plan factoriel
par(mfrow=c(1, 2))
plot.PCA(rPCA, axes=c(1,3), choix="ind", habillage=colclust, invisible="quali")
plot.PCA(rPCA, axes=c(1,3), choix="var")

# ACP sur les données centrées-réduites
rPCAnorm <- PCA(XnormplusCluster, graph=FALSE, quali.sup=colclust)

# Nuage des individus et des variables dans le premier plan factoriel
par(mfrow=c(1, 2))
plot.PCA(rPCAnorm, axes=c(1,2), choix="ind", habillage=colclust, invisible="quali")
plot.PCA(rPCAnorm, axes=c(1,2), choix="var")

# Nuage des individus et des variables dans le deuxième plan factoriel
par(mfrow=c(1, 2))
plot.PCA(rPCAnorm, axes=c(1,3), choix="ind", habillage=colclust, invisible="quali")
plot.PCA(rPCAnorm, axes=c(1,3), choix="var")

# Classification hiérarchique de Ward sur données brutes
d <- dist(X)
tree <- hclust(d^2, method="ward.D2")
par(mfrow=c(1,1))
plot(tree)

# Classification hiérarchique de Ward sur données centrées-réduites
dnorm <- dist(Xnorm)
treenorm <- hclust(dnorm^2,method="ward.D2")
plot(treenorm)

# Concatenation des données avec leur résultat de cluster
clusterW <- as.factor(cutree(tree, numcluster))
XplusClusterW <- data.frame(X, cluster=clusterW)
print(XplusClusterW)
clusternormW <- as.factor(cutree(treenorm,numcluster))
XnormplusClustW <- data.frame(Xnorm, cluster=clusternormW)
print(XnormplusClustW)

# ACP sur les données brutes
rPCAW <- PCA(XplusClusterW, scale.unit=FALSE, graph=FALSE, quali.sup=colclust)

# Nuage des individus et des variables dans le premier plan factoriel
par(mfrow=c(1, 2))
plot.PCA(rPCAW, axes=c(1,2), choix="ind", habillage=colclust, invisible="quali")
plot.PCA(rPCAW, axes=c(1,2), choix="var")

# Nuage des individus et des variables dans le deuxième plan factoriel
par(mfrow=c(1, 2))
plot.PCA(rPCAW, axes=c(1,3), choix="ind", habillage=colclust, invisible="quali")
plot.PCA(rPCAW, axes=c(1,3), choix="var")

# ACP sur les données centrées-réduites
rPCAnormW <- PCA(XnormplusClustW, scale.unit=FALSE, graph=FALSE, quali.sup=colclust)

# Nuage des individus et des variables dans le premier plan factoriel
par(mfrow=c(1, 2))
plot.PCA(rPCAnormW, axes=c(1, 2), choix="ind", habillage=colclust, invisible="quali")
plot.PCA(rPCAnormW, axes=c(1, 2), choix="var")

# Nuage des individus et des variables dans le deuxième plan factoriel
par(mfrow=c(1, 2))
plot.PCA(rPCAnormW, axes=c(1, 3), choix="ind", habillage=colclust, invisible="quali")
plot.PCA(rPCAnormW, axes=c(1, 3), choix="var")
