# Adresse du dossier où vous travaillez
setwd("/home/charles/github/CREMIfiles/IF344_Analyse/Code")
# Packages utilisés dans la suite
library("FactoMineR")

# Données sur les fromages
X <- read.table("../Data/fromage.txt", sep="", header=TRUE, row.names=1)
print(X)
