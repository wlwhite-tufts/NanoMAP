# current version of the 1st clustering step from the old pipeline
# pools assembled and filtered sequencing data from a set of samples
# moving away from this pileline, but keep around for legacy reasons

from Bio.Seq import Seq
import numpy as np
import pandas as pd
import sys
import re
import os
import shutil
import subprocess
from glob import glob
from tqdm import tqdm
import argparse
from multiprocessing import Pool

sys.path.append('/cluster/tufts/cowenlab/wwhite06/vhh/scripts/')
from python_utils import *

#get user inputs
parser = argparse.ArgumentParser(prog='Cluster Based on Immcantation annotations.',
                                 description='Collect sequences from input folder and cluster them using an agglomerative clustering method.')
#basic args
parser.add_argument('--in_dir',type=str,help='The directory that contains the assembled sequences. Looks for *db-pass.tsv, or */*db-pass.tsv')
parser.add_argument('--out_file',type=str,help='Name for the output file (can include file path).')

#translation args
parser.add_argument('--start_seq',type=str,default='LVQPGGSLRLSC',help='Amino acid sequence to scan for when finding starting point of translation.')
parser.add_argument('--end_seq',type=str,default='WGQGTQVTVSS',help='Amino acid sequence to scan for when finding ending point of translation.')
parser.add_argument('--min_start_match',type=int,default=5,help='Minimum number of amino acids matching start_seq to be considered a real sequence.')
parser.add_argument('--min_end_match',type=int,default=5,help='Minimum number of amino acids matching end_seq to be considered a real sequence.')
parser.add_argument('--max_start_scan',type=int,default=40,help='Number of N-term amino acid positions to scan when looking for start_seq.')
parser.add_argument('--max_end_scan',type=int,default=40,help='Number of C-term amino acid positions to scan when looking for end_seq.')

args = parser.parse_args()

in_dir = args.in_dir
out_file = args.out_file
out_dir = '/'.join(out_file.split('/')[:-1])+'/'

#set up for parallelization
ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
pool = Pool(processes=ncpus)
print(f'Connected to pool with {ncpus} cpus.',flush=True)

#read all files into big dataframe (this will have duplicates)
files = glob(f'{in_dir}*db-pass.tsv')
if len(files) == 0:
    print("Can't find tsv files - looking one dir deeper.")
    files = glob(f'{in_dir}*/*db-pass.tsv')
    
print('Reading the following files:')
print('\n'.join([f.split('/')[-1] for f in files]))
data = pd.concat([pd.read_csv(tsv,sep='\t') for tsv in tqdm(files)],axis=0)
#drop less useful info
data = data[["sequence", "locus", "sequence_alignment", "junction", "junction_aa", "v_call", "d_call", "j_call", "duplicate_count", 
             "fwr1", "fwr2", "fwr3", 'fwr4', "cdr1", "cdr2", "cdr3"]]

print(f'{len(data)} rows in global set after loading tsvs.', flush=True)

#define functions for aggregation
agg_funcs = {}
for c in data.columns:
    #take the first of each VHH group for descriptor columns
    if c != 'duplicate_count':
        agg_funcs[c] = lambda x: x.iloc[0]
    #sum the duplicate count
    else:
        agg_funcs[c] = np.sum

#aggregate based on sequence
data = data.groupby(by='sequence',sort=False).aggregate(func=agg_funcs)
data.reset_index(inplace=True,drop=True) #not sure why, but need drop=True here to avoid error, despite not needing this in other cases
print(f'{len(data)} unique DNA sequences.', flush=True)

#get translations based on scanning for start and end sequences
seq_info = pd.DataFrame(pool.starmap(translate_VHH_end_scan,
                                     [(seq, args.start_seq, args.end_seq, args.max_start_scan, args.max_end_scan) for seq in data['sequence']],
                                     chunksize=5000),
                        columns=['sequence','VHH','start_match','start_i','start_f','end_match','end_i'])
data = data.merge(seq_info,on='sequence',how='outer')
print(f'{len(data)} unique DNA sequences after merging translation info.', flush=True)

#get filtering statistics
has_stop = data['VHH'].map(lambda x: '*' in x)
bad_start = data['start_match'] < args.min_start_match
bad_end = data['end_match'] < args.min_end_match
print(f'{np.sum(has_stop)} ({round(np.mean(has_stop)*100,2)}%) of sequences have an early stop.', flush=True)
print(f'{np.sum(bad_start)} ({round(np.mean(bad_start)*100,2)}%) of sequences have <{args.min_start_match} matches to the start sequence.',flush=True)
print(f'{np.sum(bad_end)} ({round(np.mean(bad_end)*100,2)}%) of sequences have <{args.min_start_match} matches to the end sequence.', flush=True)

#filter out bad sequences
data = data.loc[(~has_stop)&(~bad_start)&(~bad_end),:]
print(f'{len(data)} unique DNA sequences after filtering out these errors.', flush=True)

#get duplicate count of the AA seq encoded by the DNA sequence
vhh_agg_funcs = {'VHH':lambda x: x.iloc[0], 'duplicate_count':np.sum}
vhh_agg = data[['VHH','duplicate_count']].groupby('VHH').aggregate(func=vhh_agg_funcs)
vhh_agg['VHH'] = vhh_agg.index
vhh_agg.index = range(len(vhh_agg))
print(f'{len(vhh_agg)} unique VHHs.', flush=True)
vhh_agg = vhh_agg.rename({'duplicate_count':'VHH_duplicate_count'},axis=1)

#merge vhh duplicate counts onto main df
data = data.merge(vhh_agg,on='VHH',how='outer')
print(f'{len(data)} rows after merging with AA-level data.', flush=True)

#drop sequences who's translations are observed only once across the whole dataset
data = data.loc[data['VHH_duplicate_count']>1,:]
print(f'{len(data)} unique DNA sequnces after dropping singleton AA sequences.', flush=True)
data.index = range(len(data)) #reset index

#make sequence_id column
data['sequence_id'] = data.index.map(lambda i: 'GS-'+str(i))

#run hclust in R
data.to_csv(f'{out_dir}tmp_hclust_input.csv')
code = subprocess.run(['Rscript', '/cluster/tufts/cowenlab/wwhite06/vhh/scripts/hclust_util.R',
                      f'{out_dir}tmp_hclust_input.csv', f'{out_file}', '0.75']).returncode
assert code==0

os.remove(f'{out_dir}tmp_hclust_input.csv')



    
    