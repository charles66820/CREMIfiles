# Minor introduction to research
# Analysing data and doing statistics
# Pr Sid Touati, University Côte d'Azur

library(stats)
library(graphics)
library("vioplot")

# Read the data from the first file
X <- read.csv("Data/bench1.txt", header=F)$V1
# Read the data from the second file
Y <- read.csv("Data/bench2.txt", header=F)$V1

# emprical summary
summary(X)
summary(Y)

# visualise data with boxplot
boxplot(X, Y, names=c("bench1","bench2"), col=c("blue","pink"), ylab="seconds", main="Execution times")

# visualise data with violinplot
vioplot(X, Y, names=c("bench1", "bench2"), col=c("blue","pink"), ylab="seconds", main="Execution times")

# histograms
hist(X, main="Distribution of execution times")
hist(Y, main="Distribution of execution times")

# normality check
shapiro.test(X)

# Student t-test
t.test(X, Y, "greater")
t.test(X, Y, "less")
t.test(X, Y, "two.sided")

# Kolmogorov-Smirnov
ks.test(X-median(X),Y-median(Y))

# Wilcoxon-Mann-Whitney's test
wilcox.test(X, Y, "less")
wilcox.test(X, Y, "greater")
wilcox.test(X, Y, "two.sided")
