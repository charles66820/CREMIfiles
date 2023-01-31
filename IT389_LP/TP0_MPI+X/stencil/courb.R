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

filename <- "csv/seq.csv"
dataRaw <- read.csv(filename, header=T, sep = ",")

data <- ddply(
  dataRaw,
  c("steps", "height", "width", "nbCells", "fpOpByStep"),
  summarise,
  timeInµSec_min=min(timeInµSec),
  timeInµSec_mean=mean(timeInµSec),
  timeInµSec_max=max(timeInµSec),
  gigaflops_min=min(gigaflops),
  gigaflops_mean=mean(gigaflops),
  gigaflops_max=max(gigaflops),
  cellByS_min=min(cellByS),
  cellByS_mean=mean(cellByS),
  cellByS_max=max(cellByS),
  interactions=n(),
)

SIZE <- paste(data$width, data$height, sep="_")
data <- cbind(data, SIZE)

# GFlop/s
g <- ggplot(data, aes(x=nbCells, y=gigaflops_mean))
g <- g + geom_ribbon(aes(ymin=gigaflops_min, ymax=gigaflops_max),alpha=0.2)
#g <- g + geom_errorbar(aes(ymin=gigaflops_min, ymax=gigaflops_max))
g <- g + geom_line()
g <- g + geom_point()
g <- g + labs("Perfs Seq", x="nbCells", y="GFlop/s")
plot(g)

# time(µ sec)
g <- ggplot(data, aes(x=nbCells, y=timeInµSec_mean))
g <- g + geom_ribbon(aes(ymin=timeInµSec_min, ymax=timeInµSec_max),alpha=0.2)
g <- g + geom_line()
g <- g + geom_point()
g <- g + labs("Time evolution Seq", x="nbCells", y="time(µ sec)")
plot(g)

## halos
filename <- "csv/halos.csv"
dataRaw <- read.csv(filename, header=T, sep = ",")

data <- ddply(
  dataRaw,
  c("steps", "height", "width", "tiledW", "tiledH", "nbCells", "fpOpByStep"),
  summarise,
  timeInµSec_min=min(timeInµSec),
  timeInµSec_mean=mean(timeInµSec),
  timeInµSec_max=max(timeInµSec),
  gigaflops_min=min(gigaflops),
  gigaflops_mean=mean(gigaflops),
  gigaflops_max=max(gigaflops),
  cellByS_min=min(cellByS),
  cellByS_mean=mean(cellByS),
  cellByS_max=max(cellByS),
  interactions=n(),
)

#SIZE_T <- paste(data$tiledW, data$tiledH, sep="_")
SIZE_T <- data$tiledW * data$tiledH
data <- cbind(data, size_t=order(SIZE_T, decreasing = FALSE))

#dataS = apply(data, 2, function(x) quantile(x, probs=seq(0,1, 1/10))) 

# GFlop/s
g <- ggplot(data, aes(x=size_t, y=gigaflops_mean))
#g <- g + geom_ribbon(aes(x=factor(size_t), ymin=gigaflops_min, ymax=gigaflops_max),alpha=0.2)
g <- g + geom_errorbar(aes(ymin=gigaflops_min, ymax=gigaflops_max))
g <- g + geom_line()
g <- g + geom_point()
g <- g + labs("Perfs tiled", x="size_t", y="GFlop/s")
plot(g)

# time(µ sec)
g <- ggplot(data, aes(x=size_t, y=timeInµSec_mean))
#g <- g + geom_ribbon(aes(ymin=timeInµSec_min, ymax=timeInµSec_max),alpha=0.2)
g <- g + geom_errorbar(aes(ymin=timeInµSec_min, ymax=timeInµSec_max))
g <- g + geom_line()
g <- g + geom_point()
g <- g + labs("Time evolution tiled", x="nbCells", y="time(µ sec)")
plot(g)

g <- ggplot(data, aes(x=factor(tiledW), y=factor(tiledH), fill=gigaflops_mean))
g <- g + geom_tile()
g <- g + coord_fixed()
g <- g + scale_fill_viridis(option="inferno", discrete=FALSE)
g <- g + labs(title="Test tiled", x="tiledW", y="tiledH")
plot(g)

