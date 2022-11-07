# Adresse du dossier où vous travaillez
setwd("/home/charles/github/CREMIfiles/IF344_Analyse/Code")

# Packages utilisés dans la suite
library("FactoMineR") # install.packages("FactoMineR")
library(PCAmixdata) # install.packages("PCAmixdata")
library(GGally) # install.packages("GGally")

# Chargement des données
load("../../Data/eaux.RData")
# Affichage des données
print(data, digits=4)

# Calcul de la moyenne et de l’écart type des variables
mean <- apply(data, 2, mean)
std <- apply(data, 2, sd) #standard deviation
stat <- rbind(mean, std)
# Affichage
print(stat, digits=4)

# Création des données centrées ...
datanorm <- sweep(data, 2, mean, "-")
# ... et réduites
datanorm <- sweep(datanorm, 2, std, "/")
# Affichage des données centrées - réduites
print(datanorm, digits=4)

# Visualisation des données en description bivariée
pairs(data[,1:5])
# Afficher la matrice de corrélation
ggcorr(data[,1:5])
# Aller encore plus loin avec ggpairs
ggpairs(data[,1:5])

# Matrice des distances entre les individus
dist(data)
# Corrélation entre les variables
cor(data[,1:5])

# Analyse en composantes principales sur les données d’origine
# (scale.unit=FALSE)
res <- PCA(data, graph=FALSE, scale.unit=FALSE)
# Figure individus
plot(res, choix="ind", cex=1.5, title="")
# Figure variables
plot(res, choix="var", cex=1.5, title="")

# Analyse en composantes principales sur les données centrées-réduites
# (par défaut: scale.unit=TRUE)
resnorm <- PCA(data, graph=FALSE)
# Figure individus
plot(resnorm, choix="ind", cex=1.5, title="")
# Figure variables
plot(resnorm, choix="var", cex=1.5, title="")

# Inertie (variance) des composantes principales
resnorm$eig
barplot(resnorm$eig[,1])

# Projection des individus
resnorm$ind$cos2
# Somme avec les 2 premières
resnorm$ind$cos2[,1]+resnorm$ind$cos2[,2]
# Et les 3 premières ?
resnorm$ind$cos2[,1]+resnorm$ind$cos2[,2]+resnorm$ind$cos2[,3]

# Contribution des individus
resnorm$ind$contrib

# Projection des variables
resnorm$var$cos2
# Somme avec les 2 premières
resnorm$var$cos2[,1]+resnorm$var$cos2[,2]
# Et les 3 ?
resnorm$var$cos2[,1]+resnorm$var$cos2[,2]+resnorm$var$cos2[,3]
# Contribution des variables
resnorm$var$contrib
