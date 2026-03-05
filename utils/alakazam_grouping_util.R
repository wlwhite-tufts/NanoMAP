library('magrittr')
library('dplyr')
library('foreach')
library('doParallel')
library('alakazam')

args = commandArgs(trailingOnly=TRUE)
input_file <- args[1]
output_file <- args[2]
junc_len_col <- args[3]

dat <- read.csv(input_file, header = T, stringsAsFactors = F)

dat <- alakazam::groupGenes(dat,
                         v_call = 'v_call',
                         j_call = 'j_call',
                         junc_len = junc_len_col,
                         first = FALSE)
                         
write.csv(dat, file = output_file, row.names = F)
