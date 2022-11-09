library(dplyr)

data <- read.table("data.csv", sep=",", header=TRUE)

#plot(data$time~data$size, xlab="size", ylab="time")

rank0_data <- filter(data, rank == 0)
rank1_data <- filter(data, rank == 1)

plot(rank1_data$time~rank1_data$size, xlab="size", ylab="time")
matlines(rank0_data$size, cbind(rank0_data$time, rank1_data$time), lty=c(2, 3), col=c("green3", "blue"), lwd=c(2, 2))
legend("topleft", lty=c(2, 3), lwd=c(2, 2), c("rank0", "rank1"), col=c("green3", "blue"))

