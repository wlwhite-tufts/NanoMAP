# script to assemble and filter the raw sequencing data
# produces files that are used by both clustering methods

## ###
## VHH-Assembly.sh
## Relies heavily on runalignmentpipeline.sh
## (Written by FAJO, of the Cowen Lab)
## ###

# Define environmental variables

PRESTO_BIN=/cluster/tufts/cowenlab/immcantation_suite/presto/bin
CHANGEO_BIN=/cluster/tufts/cowenlab/immcantation_suite/changeo/bin
NCBI_DIR=/cluster/tufts/cowenlab/wwhite06/packages/igblast_v3/ncbi-igblast-1.22.0

IN_DIR="$1"
OUT_DIR="$2"
R1_PRIMERS="$3"
R2_PRIMERS="$4"

if [ ! -d "$OUT_DIR" ]; then
    mkdir "$OUT_DIR"
fi


# Iterate over each file in the assembly folder
# Pipe through each unique file name after trimming everyhing after the "_L"
for sample in $(ls "$IN_DIR" | awk -F'_L0' '{print $1}' | uniq); do
    echo $sample
    # Create all necessary file names
    SEQ_RONE="$IN_DIR"/${sample}_L001_R1_001.fastq
    SEQ_RTWO="$IN_DIR"/${sample}_L001_R2_001.fastq
    echo 'Processing the following files: '
    echo $SEQ_RONE
    echo $SEQ_RTWO
    SEQ_QUAL="$OUT_DIR"/${sample}_assemble-pass.fastq
    echo $SEQ_QUAL
    SEQ_MPV="$OUT_DIR"/${sample}_quality-pass.fastq
    echo $SEQ_MPV
    SEQ_MPC="$OUT_DIR"/${sample}_MPV_primers-pass.fastq
    echo $SEQ_MPC
    SEQ_COL="$OUT_DIR"/${sample}_MPC_primers-pass.fastq
    echo $SEQ_COL
    SEQ_DUP="$OUT_DIR"/${sample}_collapse-unique.fastq
    echo $SEQ_DUP
    SEQ_FASTA="$OUT_DIR"/${sample}_atleast-1.fasta
    echo $SEQ_FASTA
    SEQ_FMT7="$OUT_DIR"/${sample}_atleast-1.fmt7
    echo $SEQ_FMT7
    # STEP 1: Alignment
    ${PRESTO_BIN}/AssemblePairs.py align -1 $SEQ_RTWO -2 $SEQ_RONE --coord illumina --rc tail --maxerror 0.2 --outname $sample --outdir "$OUT_DIR"
    # STEP 2: Quality Filtering
    ${PRESTO_BIN}/FilterSeq.py quality -s $SEQ_QUAL -q 20 --outname $sample --outdir "$OUT_DIR"
    # STEP 3: Variable Primer Pass
    ${PRESTO_BIN}/MaskPrimers.py score -s $SEQ_MPV -p "$R2_PRIMERS" --start 0 --mode cut --pf VPRIMER --outname ${sample}_MPV --outdir "$OUT_DIR"
    # STEP 4: Constant Primer Pass
    ${PRESTO_BIN}/MaskPrimers.py score -s $SEQ_MPC -p "$R1_PRIMERS" --start 0 --mode cut --revpr --pf CPRIMER --outname ${sample}_MPC --outdir "$OUT_DIR"
    # STEP 5: Deduplicate
    ${PRESTO_BIN}/CollapseSeq.py -s $SEQ_COL -n 20 --inner --uf CPRIMER --cf VPRIMER --act set --outname $sample --outdir "$OUT_DIR"
    # STEP 5: Split by duplicate count
    ${PRESTO_BIN}/SplitSeq.py group -s $SEQ_DUP -f DUPCOUNT --num 1 --outname $sample --outdir "$OUT_DIR"
    # Perform this same operation again, but this time we want a .fasta output file
    ${PRESTO_BIN}/SplitSeq.py group -s $SEQ_DUP -f DUPCOUNT --num 1 --fasta --outname $sample --outdir "$OUT_DIR"
    # Move to directory to see if that makes it work
    cd $NCBI_DIR

    # Now we want to use ncbi
    ${NCBI_DIR}/bin/igblastn -germline_db_V  $NCBI_DIR/internal_data/alpaca/alpaca_V -germline_db_D $NCBI_DIR/internal_data/alpaca/alpaca_D -germline_db_J $NCBI_DIR/internal_data/alpaca/alpaca_J -domain_system imgt -ig_seqtype Ig -organism alpaca -auxiliary_data $NCBI_DIR/optional_file/alpaca_gl.aux -outfmt '7 std qseq sseq btop' -query $SEQ_FASTA -out $SEQ_FMT7
    # Now we build the database
    ${CHANGEO_BIN}/MakeDb.py igblast -i $SEQ_FMT7 -s $SEQ_FASTA -r $NCBI_DIR/database/alpaca/IG_DNA/imgt_gapped_alpaca_IGHV.fasta  $NCBI_DIR/database/alpaca/IG_DNA/imgt_gapped_alpaca_IGHD.fasta $NCBI_DIR/database/alpaca/IG_DNA/imgt_gapped_alpaca_IGHJ.fasta --outdir "$OUT_DIR" --outname $sample --failed --log "$OUT_DIR"/MakeDb_${sample}.log --format airr --extended

done
