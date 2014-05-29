#!/usr/bin/Rscript
filename <- "wikipedia_data.png"
data <- read.table("data/wikipedia.data", header=FALSE, sep="," ,comment.char="#")
png(filename)
plot(data$V1[1:156],data$V2[1:156],col='red',xlab="X",ylab="Y",main="Original Data")
points(data$V1[1:30],data$V2[1:30],col='red')
points(data$V1[31:90],data$V2[31:90],col='green')
points(data$V1[91:156],data$V2[91:156],col='blue')
x <-dev.off()

filename <- "wikipedia_RBF_transformed.png"
data <- read.table("data/transformed_RBF_data.csv", header=FALSE, sep="," ,comment.char="#")
png(filename)
plot(data$V1[1:156],data$V2[1:156],col='red',xlab="PC1",ylab="PC2",main="RBF transformed Data - PC1 & PC2")
points(data$V1[1:30],data$V2[1:30],col='red')
points(data$V1[31:90],data$V2[31:90],col='green')
points(data$V1[91:156],data$V2[91:156],col='blue')
x <-dev.off()

filename <- "wikipedia_Polynomial_transformed.png"
data <- read.table("data/transformed_Polynomial_data.csv", header=FALSE, sep="," ,comment.char="#")
png(filename)
plot(data$V1[1:156],data$V2[1:156],col='red',xlab="PC1",ylab="PC2",main="Polynomial transformed Data - PC1 & PC2")
points(data$V1[1:30],data$V2[1:30],col='red')
points(data$V1[31:90],data$V2[31:90],col='green')
points(data$V1[91:156],data$V2[91:156],col='blue')
x <-dev.off()