options(repos = c(CRAN = "https://cran.rstudio.com"))

install.packages('yaml')
install.packages("this.path")

install.packages("BiocManager",repos = "https://cloud.r-project.org")
BiocManager::install('scoper')
BiocManager::install('alakazam')
