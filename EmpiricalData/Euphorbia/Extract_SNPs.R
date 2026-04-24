library(adegenet)

files <- list.files(path="../Original_Fastas", pattern="*.fas", full.names=TRUE, recursive=FALSE)
for (i in files) {
  a=fasta2genlight(i)
  write.csv(a, file=paste(i,".csv"))
}

genot <- list.files(path=".", pattern="*.csv", full.names=TRUE, recursive=FALSE)
b=data.frame()
for (i in genot) {
  df=read.csv(i)
  chosen_SNP <- order(colSums(is.na(df[, -1])))[1] + 1
  if (nrow(b) == 0) {
    b <- df[, chosen_SNP, drop = FALSE] # First iteration: 'b' becomes the new column
  } else {
    b <- cbind(b, chosen_SNP) # Later iterations: append the new column
  }
}

write.csv(b, "input_SNPs.csv")
