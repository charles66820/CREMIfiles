library(plyr)
library(dplyr)
library(ggplot2)
library(viridis)

# Clean
rm(list=ls(all=TRUE))
graphics.off()

# Set Env
currentScriptDir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(currentScriptDir)

filename <- "seq.csv"
data <- read.csv(filename, header=T, sep = ";")

g <- ggplot(data, aes(x=size, y=flops))
g <- g + geom_line()
g <- g + geom_point()
g <- g + labs("Perf", x="Size", y="GFlop/s")
plot(g)

