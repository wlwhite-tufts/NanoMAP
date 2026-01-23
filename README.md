<img width="1056" height="380" alt="NanoMAP_logo" src="https://github.com/user-attachments/assets/8d291278-0edb-4222-9c0e-359f3bb51860" />
This repository contains code for processing and analyzing VHH (nanobody) sequences using NanoMAP.

# Installation
## Setup the conda environment
1. Navigate into the directory where you've cloned this repo and run the following commands:
```
conda env create -f data/nanomap_env.yml
conda activate nanomap
```
2. Install ANARCI, following the instructions in their GitHub repo: https://github.com/oxpig/ANARCI
3. Install R, following the instructions here: https://www.r-project.org/
4. Install singularity, following the instructions here: https://docs.sylabs.io/guides/3.0/user-guide/installation.html#installation

## Install IgBLAST (with our updated alpaca V/J segments)
Naviagte to a directory that you want to install IgBLAST in and run the following commands to download the package.
```
wget https://ftp.ncbi.nih.gov/blast/executables/igblast/release/LATEST/ncbi-igblast-1.22.0-x64-linux.tar.gz
tar -xzvf ncbi-igblast-1.22.0-x64-linux.tar.gz
rm ncbi-igblast-1.22.0-x64-linux.tar.gz
```
Then create some internal folders and copy our extended V and J segment lists into the IgBLAST directory:
```
mkdir ncbi-igblast-1.22.0/database
mkdir ncbi-igblast-1.22.0/database/alpaca
mkdir ncbi-igblast-1.22.0/database/alpaca_clean

cp -r <your NanoMAP path>/data/IgBLAST_db/* ncbi-igblast-1.22.0/database/alpaca/
```
Replace `<your NanoMAP path>` with the path to the directory where you cloned this repo.\
In addition to FASTA files for the V, D, and J segments, this copies over the alpaca_gl.aux and alpaca.ndm.imgt files which are used to supplement the FASTA files while utilizing the database.\
\
To parse through the sequencing data from the `alpaca` directory and convert it into an IgBLAST acceptable format in the `alpaca_clean` directory, run the following commands for the V, D, and J segments:
```
ncbi-igblast-1.22.0/bin/edit_imgt_file.pl ncbi-igblast-1.22.0/database/alpaca/IG_DNA/imgt_gapped_alpaca_IGHV.fasta > ncbi-igblast-1.22.0/database/alpaca_clean/alpaca_V
ncbi-igblast-1.22.0/bin/edit_imgt_file.pl ncbi-igblast-1.22.0/database/alpaca/IG_DNA/imgt_gapped_alpaca_IGHD.fasta > ncbi-igblast-1.22.0/database/alpaca_clean/alpaca_D
ncbi-igblast-1.22.0/bin/edit_imgt_file.pl ncbi-igblast-1.22.0/database/alpaca/IG_DNA/imgt_gapped_alpaca_IGHJ.fasta > ncbi-igblast-1.22.0/database/alpaca_clean/alpaca_J

ncbi-igblast-1.22.0/bin/makeblastdb -parse_seqids -dbtype nucl -in ncbi-igblast-1.22.0/database/alpaca_clean/alpaca_V
ncbi-igblast-1.22.0/bin/makeblastdb -parse_seqids -dbtype nucl -in ncbi-igblast-1.22.0/database/alpaca_clean/alpaca_D
ncbi-igblast-1.22.0/bin/makeblastdb -parse_seqids -dbtype nucl -in ncbi-igblast-1.22.0/database/alpaca_clean/alpaca_J
```
Then move and rename the files to match what IgBLAST expects:
```
cp -r ncbi-igblast-1.22.0/database/alpaca_clean ncbi-igblast-1.22.0/internal_data/
mv ncbi-igblast-1.22.0/internal_data/alpaca_clean ncbi-igblast-1.22.0/internal_data/alpaca

mv ncbi-igblast-1.22.0/database/alpaca/alpaca.ndm.imgt ncbi-igblast-1.22.0/internal_data/alpaca/
mv ncbi-igblast-1.22.0/database/alpaca/alpaca_gl.aux ncbi-igblast-1.22.0/optional_file/
```

## Download MMseqs2 and GFold sif files
Navigate into the directory where you've cloned this repo and run the following commands:
```
mkdir data/sif
cd data/sif
singularity pull https://depot.galaxyproject.org/singularity/mmseqs2:17.b804f--hd6d6fdc_1
mv mmseqs2:17.b804f--hd6d6fdc_1 mmseqs2-17.b804f.sif
singularity pull https://depot.galaxyproject.org/singularity/gfold_1.1.4--gsl1.16_1
mv gfold_1.1.4--gsl1.16_1 gfold-1.1.4.sif
```

## Input your local paths
Edit the `data/user_data.py` file to match the locations on your machine where you cloned this repo and downloaded the sif files.
If you are going to be consistently using the same primers to amplify your cDNA for sequencing, create one fasta file for the forward primer(s), and one for the reverse primer(s) and update the primer section of `data/user_data.py` to provide the locations of those fasta files.

## test that everything has been installed correctly
Run the following commands, replacing `<your NanoMAP path>` with the directory where you cloned this repo.
```
cd <your NanoMAP path>/test
conda activate nanomap
./run_test.sh
```
If the installation was successful, this script will run for ~10-15min without errors, and produce the following files:
```
assembly_cmds.sh
assembly/test1_db-pass.tsv
assembly/test2_db-pass.tsv
assembly/test3_db-pass.tsv
assembly/test4_db-pass.tsv

metaclust.csv
metaclust_tabulated.csv
metaclust_gfold_vhh.csv
metaclust_gfold_fam.csv

sil_score.csv
pheno_score.csv
```

# NanoMAP Meta-clustering Pipeline
This pipeline starts with a group of fastq(.gz) files resulting from a paired-end sequencing run and produces clustered sequences with enrichment values calculated per cluster and per sequence. Each pair of fastq files (forward and reverse reads) should correspond to a single sample.

## Pre-processing
First, collect all your fastq files into a single directory, and make sure their names take the following format:\
`{sample_id}_{read_id}.fastq`\
where `sample_id` is a unique identifier for the sample, and `read_id` indicates whether the fastq file is a forward or reverse read.\
Then split the fastq files into batches to process in parallel:
```
cd <directory where you want to run your pre-processing, clustering, and enrichment calculations>
python <path to NanoMAP>/processing/make_fastq_batches.py --in_dir <folder with fastq files> --cmd_file assembly_cmds.sh --N 2 --read_ids <your forward read_id> <your reverse read ID>
```
Before running the command, fill in the `<your text here>` fields and change the value of the `--N` flag depending on how many samples you want to include in each batch. Budget about 1min per 3k reads in the batch, counting each paired-end read as one.\
\
Next, run the assembly and filtering commands in `assembly_cmds.sh`, or submit them as an array job with SLURM. Each line in the `assembly_cmds.sh` file is a command to process a single batch of fastq files, and can be run independently of the other commands in the file. If any of the runs are interrupted in the middle, the corresponding command can simply be re-run and will pick up where it left off.\
\
Once all the assembly/filtering commands are finished running, you should have a directory called `assembly` containing two types of files:\
`{sample_id}_db-pass.tsv` - these are the assembled and filtered reads, there should be one per sample\
`{sample_id}_chkpt.txt` - these are no longer necessary and can be deleted

## Clustering
After completing the pre-processing, run the following command to cluster all the sequences in your dataset:
```
cd <directory where you ran the make_fastq_batches.py script>
python <path to NanoMAP>/processing/metacluster.py --in_dir assembly --out_file metaclustered_data.csv
```
Before running the command, fill in the `<your text here>` fields and change the --out_file flag if you want a different name for the clustered output file.\
This script is fairly compute-intensive. For our datasets, the following computational resources were required:\
Schistosomes (4.0M reads, 195k unique): 40 cores, 42G memory, 4h:05m wall-clock time\
CoV (7.2M reads, 326k unique): 40 cores, 47G memory, 7h:13m wall-clock time\
BoNT/A (14.1M reads, 582k unique): 40 cores, 131G memory, 11h:30m wall-clock time\
Exact runtimes and memory requirements will depend on the size and diversity of your data, but these can be useful benchmarks when deciding how many resources to allocate for your own clustering.\
\
The result of this step is a single large csv file that lists every unique DNA sequence in the dataset along with some annotations including the cluster labels from the meta-clustering, CDR sequences, and V/D/J segment calls.

## Enrichment calculations
While your clustering is running, you can prepare the necessary files to describe the enrichment calculations you want to perform. You will need two files: `sample_info.csv` and `enrich_pairs.csv` which contain information about each sample, and each pair of samples you want to compare, respectively. These files should be placed in the same directory as the `assembly` directory.
### sample_info file
Each row in this file corresponds to a sample that you want to use in enrichment analysis. The file must contain the following two columns:\
`name` - a unique and descriptive, but ideally compact identifier for the sample\
`file` - the full path to the `{sample_id}_db-pass.tsv` file corresponding to the sample
### enrich_pairs file
Each row in this file corresponds to an enrichment calculation you want to do. The file must contain the following three columns:\
`name` - a unique identifier for the enrichment ratio you're calculating\
`initial` - the sample name (matching the `name` column of `sample_info.csv`) to use as the denominator of the enrichment ratio. You can include any number of sample names here, each separated by a `+` (i.e. `replicate1+replicae2+replicate3`).\
`final` - the sample name to use as the numerator of enrichment calculations, following the same format as `initial`
### running enrichment calculations
Once your clustering is done and you've created `sample_info.csv` and `enrich_pairs.csv`, you can run the enrichment analysis:
```
cd <directory where you ran the make_fastq_batches.py script>
python <path to NanoMAP>/analysis/enrichment.py --in_file metaclustered_data.csv --sample_file sample_info.csv --pair_file enrich_pairs.csv
```
The result will be two new csv files:
`metaclustered_data_gfold_vhh.csv` - each row corresponds to a distinct amino acid sequence\
`metaclustered_data_gfold_fam.csv` - each row corresponds to a distinct cluster (a.k.a. clonal family)\
Both files will have the same columns, which include all of the columns that were in `metaclustered_data.csv` plus three new sets of columns:
1. Raw reads - there will be one of these columns per row in `sample_info.csv`, with the raw read counts for each sequence (or cluster) in each sample. These column names will exactly match the values in the `name` column of `sample_info.csv`.
2. enrichments - there will be one of these columns per row in `enrich_pairs.csv`, with the log of the ratio of the read counts for the samples described by that row. These column names will look like `enrich_{name}` where `name` matches the `name` column of `enrich_pairs.csv`.
3. gfold - there will be one of these columns per row in `enrich_pairs.csv`, with the gfold value comparing the samples described by that row. These column names will look like `gfold01_{name}` where `name` matches the `name` column of `enrich_pairs.csv`.

## Scoring (optional)
To evaluate the quality of your clustering you can calculate the scores below.

### Silhouette score
The following code approximates the silhouette score as described in the NanoMAP manuscript. The score varies from -1 (for very bad clusterings), to 1 (for clusterings where each member of each cluster is distance 0 from all other members of the cluster and sitance 1 from all other sequences).\
You can calculate the silhouette score with the following command:
```
cd <directory where you clustering results are>
python <path to NanoMAP>/analysis/silhouette_score.py --in_file metaclustered_data.csv --out_file sil_score.csv --label_cols meta_clone_id_single_0.25
```

### Phenotypic quality score
The following code approximates the phenotypuc quality score as described in the NanoMAP manuscript (without the final rescaling step). The score has no theoretical bounds, but higher scores are better.\
IMPORTANT: The phenotypic quality score is dependent on the size of the dataset, with smaller datasets producing lower scores even when the quality of the clustering is held fixed.\
You can calculate the phenotypic quality score with the following command:
```
cd <directory where you clustering results are>
python <path to NanoMAP>/analysis/phenotype_quality.py --in_file metaclustered_data_gfold_vhh.csv --out_file phen_score.csv --label_cols meta_clone_id_single_0.25 --phenotype_cols <your columns here>
```
Before running, make sure to set the `--phenotype_cols` flag to indicate which columns in your clustering file contain binding data. These columns could be any subset of the `enrich_{name}` or `gfold01_{name}` columns produced by the `enrichment.py` script.
