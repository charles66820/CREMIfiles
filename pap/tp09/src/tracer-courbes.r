

args = commandArgs(trailingOnly=TRUE)


nbFichiers = length(args) 

if (nbFichiers != 1) {
  stop("Un unique fichier svp", call.=FALSE)
}


data <- read.table(args[1])


library("plyr")
library("ggplot2")


data_summary <- function(data, varname, groupnames){
    require(plyr)
    summary_func <- function(x, col){
        c(mean = mean(x[[col]], na.rm=TRUE),
        sd = sd(x[[col]], na.rm=TRUE))
    }
    data_sum<-ddply(data, groupnames, .fun=summary_func,
    varname)
    data_sum <- rename(data_sum, c("mean" = varname))
    return(data_sum)
}

data_stat <- data_summary(data,"V3", groupnames=c("V1","V2"))


courbes <- ggplot(data=data_stat, aes(x=log(V2), y=V3,  color=factor(V1))) + geom_line()  + scale_y_log10()   + labs(color="taille") + geom_errorbar(aes( ymin=V3-sd, ymax=V3+sd), width=0.2, size=1)

courbes