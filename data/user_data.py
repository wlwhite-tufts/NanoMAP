import os

presto_path = '/cluster/tufts/cowenlab/immcantation_suite/presto/bin'
changeo_path = '/cluster/tufts/cowenlab/immcantation_suite/changeo/bin'
igblast_path = '/cluster/tufts/cowenlab/wwhite06/packages/igblast_v3/ncbi-igblast-1.22.0'

repo_path = os.path.dirname(os.path.abspath(__file__))
repo_path = '/'.join(repo_path.split('/')[:-1])
gfold_sif = f'{repo_path}/sif/gfold-1.1.4.sif'
mmseqs_sif = f'{repo_path}/sif/mmseqs2-17.b804f.sif'

R1_primer_file = f'{repo_path}/data/primers/AlpVHH_R1_primers_05Feb24.fasta'
R2_primer_file = f'{repo_path}/data/primers/AlpVHH_R2_primers_05Feb24.fasta'