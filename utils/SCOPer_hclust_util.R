library('magrittr')
library('dplyr')
library('foreach')
library('doParallel')
library('alakazam')
library('scoper')

args = commandArgs(trailingOnly=TRUE)
input_file <- args[1]
output_file <- args[2]
cutoff <- args[3]
linkage <- args[4]
junction_col <- args[5]
cluster_col <- args[6]

dat <- read.csv(input_file, header = T, stringsAsFactors = F)

cat("Inferring Clones:\n")
system.time(
# Clonal assignment using hierarchical clustering
db <- hierarchicalClones(dat, 
                             threshold = cutoff, 
                             linkage = linkage,
                             method = 'nt',
                             v_call = "v_call", 
                             j_call = "j_call",
                             junction = junction_col,
                             clone = cluster_col,
                             verbose = T,
                             summarize_clones = FALSE)
)

write.csv(db, file = output_file, row.names = F)


