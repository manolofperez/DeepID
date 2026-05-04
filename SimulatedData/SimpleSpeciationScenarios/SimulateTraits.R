#!/usr/bin/env Rscript

# Script By: Manolo Perez
# A script to simulate continuous traits on several newick trees saved in a file.

# To install the required packages, uncomment the lines below
# install.packages("ape",repos="https://cloud.r-project.org")
# install.packages("geiger",repos="https://cloud.r-project.org")
# install.packages("phytools",repos="https://cloud.r-project.org")
# install.packages("reticulate")
# install.packages(c( "foreach", "doParallel") )

# Load the required packages.
library(ape)
library(geiger)
library(phytools)
library(foreach)
library(doParallel)
library(reticulate)
# import the NumPy function from the package reticulate (interface with Python)
np <- import("numpy",convert=F)

#setup parallel backend to use many processors.
cl <- makeCluster(5) #This is to use 10 cores, adjust accordingly (leaving at least one free core). Alternatively one can use detectCores() function.
registerDoParallel(cl)

#Create a folder for storing the simulated traits.
dir.create("traits")

# Number of total traits for each simulation.
nsims <- 100 
# Read the total number of trees (here it was 100 trees for each simulated data set).
ntrees <- np$load("trees.npz")

# Load trees as a matrix, with the number of rows equal to the number of simulated datasets and 100 columns (trees per simulated data set).
trees<-matrix(ntrees$f[["trees"]], ncol=100,byrow=TRUE)

### Simulate traits ###
# BM
traits_BM={}

# start a parallelized loop that reads the trees in the matrix and assigns coordinates (i,j) to each tree.
traits_BM<-foreach(i=1:nrow(trees),.combine=rbind,.packages = c("phytools","ape","geiger")) %:%
  foreach(j=1:nsims,.combine=cbind,.packages = c("phytools","ape","geiger")) %dopar% {
    # read the current tree.
    tree <- read.tree(text=trees[i,j])
    # each diploid individual has two tips in the tree, so we need to drop one of them (the second) to correctly simulate the traits.
    tree <-drop.tip(tree, as.character(seq(0,length(tree$tip.label),by=2)))
    # simulate traits using the topology.
    sims <- fastBM(tree, a = 0, sig2 = 0.06, nsim = 1) 		
    #order the traits individuals according to the SNPs order.
    ordered_sims <- sims[order(as.numeric(names(sims)))]
    return(ordered_sims)
}	
# save the output.
write.table(format(as.matrix(traits_BM), width = 6),"./traits/traits_BM.txt",row.names=F,col.names=F,quote = F)

# OU
traits_OU={}

# start a parallelized loop that reads the trees in the matrix and assigns coordinates (i,j) to each tree.
traits_OU<-foreach(i=1:nrow(trees),.combine=rbind,.packages = c("phytools","ape","geiger")) %:%
  foreach(j=1:nsims,.combine=cbind,.packages = c("phytools","ape","geiger")) %dopar% {
    # read the current tree.
    tree <- read.tree(text=trees[i,j])
    # each diploid individual has two tips in the tree, so we need to drop one of them (the second) to correctly simulate the traits.
    tree <-drop.tip(tree, as.character(seq(0,length(tree$tip.label),by=2)))
    # simulate traits using the topology.
    sims <- fastBM(tree, a = 0, sig2 = 0.06, alpha = 0.2, theta = 0, nsim = 1) 		
    #order the traits individuals according to the SNPs order.
    ordered_sims <- sims[order(as.numeric(names(sims)))]
    return(ordered_sims)		
}
# save the output.
write.table(format(as.matrix(traits_OU), width = 6, flag = "0"),"./traits/traits_OU.txt",row.names=F,col.names=F,quote = F)

# Discrete trait - multivariate (3 states)
traits_disc={}

# start a parallelized loop that reads the trees in the matrix and assigns coordinates (i,j) to each tree.
traits_disc<-foreach(i=1:nrow(trees),.combine=rbind,.packages = c("phytools","ape","geiger")) %:%
  foreach(j=1:nsims,.combine=cbind,.packages = c("phytools","ape","geiger")) %dopar% {
    # read the current tree.
    tree <- read.tree(text=trees[i,j])
    # each diploid individual has two tips in the tree, so we need to drop one of them (the second) to correctly simulate the traits.
    tree <-drop.tip(tree, as.character(seq(0,length(tree$tip.label),by=2)))
    # simulate traits using the topology.
		statecols <- vector()
			qq <- matrix(c(-1, .5, .5, .5, -1, .5, .5, .5, -1),3,3)
			msims <- sim.history(tree, qq, message=FALSE)
			statecols <- c(statecols, msims$states)	
		# store the simulated values in a matrix.
		sims <- matrix(unlist(statecols), ncol=1)		
    #order the traits individuals according to the SNPs order.
		rownames(sims) <- tree$tip.label	
		ordered_sims <- sims[order(as.numeric(row.names(sims))), ]						
		return(ordered_sims)
}	
# save the output.
write.table(as.matrix(traits_disc),"./traits/traits_disc.txt",row.names=F,col.names=F,quote=F)
# Stop the parallel process.
stopImplicitCluster()	
