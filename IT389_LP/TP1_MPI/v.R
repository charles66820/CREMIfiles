library(dplyr)

data <- read.table("data.csv", sep=",", header=TRUE)

rank0_data <- filter(data, rank == 0)
rank1_data <- filter(data, rank == 1)

plot(rank1_data$time~rank0_data$size, type="l", col=c("blue"), lty=c(3), lwd=2, xlab="size", ylab="time")
lines(rank0_data$size, rank0_data$time, col=c("green3"), lty=c(2), lwd=2)
legend("topleft", lty=c(2, 3), lwd=c(2, 2), c("rank0", "rank1"), col=c("green3", "blue"))
