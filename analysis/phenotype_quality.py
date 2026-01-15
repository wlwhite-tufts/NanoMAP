import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ranksums
from tqdm import tqdm
import argparse
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
import os

def get_args():
    #get user inputs
    parser = argparse.ArgumentParser(prog='Phenotype based clustering quality score.',
                                     description='Score clusterings based on the difference in phenotype between and within clusters.')
    #I/O args
    parser.add_argument('--in_file',type=str,help='csv file containing sequence and cluster label info')
    parser.add_argument('--in_file_header',default=0,type=int,help='number of header rows in input file to skip')
    parser.add_argument('--out_file',type=str,help='Name for the output file (can include file path).')
    parser.add_argument('--label_cols',type=str,nargs='*',help='Which column(s) contain the cluster labels to be evaluated')
    parser.add_argument('--drop_duplicates',type=str,default='VHH',help='Which column to use to drop duplicate entries before scoring. Set to "none" to skip this step.')
    
    #phenotype args
    parser.add_argument('--phenotype_cols',type=str,nargs='*', help='Which columns to use for phenotype data when calculating pairwise distances.')
    parser.add_argument('--dist_metric',type=str,default='correlation',help="Distance metric to use when clustering by phenotype. Can be any metric allowed by scipy's pdist.")
    parser.add_argument('--max_log_p',type=float,default=750,help="Ceiling for -log(p-value) scores for each sequence.")
    
    return parser.parse_args()
    
def ranksum_score_single(dists_within, dists_between, max_log_p):

    n_within = len(dists_within)
    n_between = len(dists_between)
    
    # if too few within-cluster dists, pad with zeros or existing value
    if (n_within==0):
        dists_within = [0,0]
    elif (n_within==1):
        dists_within = [dists_within[0]]*2
        
    #if too few betewen-cluster dists, this is a bad sequence and should be ignored
    if (n_between<2):
        return np.nan
        
    p_val = ranksums(dists_within,dists_between,
                    alternative='less',
                    nan_policy='omit').statistic#pvalue
    # if p_val == 0:
    #     score = max_log_p
    # else:
    #     score = min(-np.log(p_val),max_log_p)
    
    # return score/max_log_p
    return p_val
    
def ranksum_score_multilabel(idx, N, N_labels, N_pheno, dist_metric, max_log_p, rand):
    
    label_shm = SharedMemory(name=f'labels_{rand}')
    labels = np.ndarray((N,N_labels), dtype=np.double, buffer=label_shm.buf)
    
    pheno_shm = SharedMemory(name=f'pheno_{rand}')
    pheno = np.ndarray((N,N_pheno), dtype=np.double, buffer=pheno_shm.buf)
    
    pheno_row = pheno[idx,:]
    dists = cdist(np.expand_dims(pheno_row,0),pheno,metric=dist_metric).squeeze()
                  
    not_nan = ~np.isnan(dists)
    scores = []
    for c in range(labels.shape[1]):
        in_clust = labels[:,c]==labels[idx,c]
        dists_within = dists[in_clust&not_nan]
        dists_between = dists[(~in_clust)&not_nan]
        
        scores.append(ranksum_score_single(dists_within, dists_between, max_log_p))
        
    label_shm.close()
    pheno_shm.close()
        
    print(idx,flush=True)
        
    return scores
    
def ranksum_score_all(data, label_cols, phenotype_cols, dist_metric, max_log_p):
    
    N = len(data)
    N_labels = len(label_cols)
    N_pheno = len(phenotype_cols)
    
    rand = np.random.randint(0,high=1e6)
    
    label_mem = SharedMemory(name=f'labels_{rand}', create=True, size=8*N*N_labels)
    labels = np.ndarray((N,N_labels), dtype=np.double, buffer=label_mem.buf)
    labels[:] = data[label_cols].to_numpy()[:]
    
    pheno_mem = SharedMemory(name=f'pheno_{rand}', create=True, size=8*N*N_pheno)
    pheno = np.ndarray((N,N_pheno), dtype=np.double, buffer=pheno_mem.buf)
    pheno[:] = data[phenotype_cols].to_numpy()[:]
    
    ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    pool = Pool(processes=ncpus)
    print(f'Connected to pool with {ncpus} cpus.',flush=True)
    
    scores = np.array(pool.starmap(ranksum_score_multilabel, ([idx, N, N_labels, N_pheno, dist_metric, max_log_p, rand] for idx in range(N))))
    scores = np.nanmean(scores,axis=0)
    scores = {col:scores[c] for c,col in enumerate(label_cols)}
    
    pool.close()
    label_mem.close()
    label_mem.unlink()
    pheno_mem.close()
    pheno_mem.unlink()
    
    return scores
    
def fix_label_type(obj):
    if type(obj)==str and obj.startswith('GS-'):
        return int(obj.split('-')[1])
    return obj
    
if __name__ == '__main__':
    
    args = get_args()
    
    data = pd.read_csv(args.in_file,header=args.in_file_header)
    if args.drop_duplicates.lower() != 'none':
        data = data.drop_duplicates(args.drop_duplicates)
        
    for col in args.label_cols:
        data[col] = data[col].map(fix_label_type) 
        
    scores = ranksum_score_all(data, args.label_cols, args.phenotype_cols, args.dist_metric, args.max_log_p)

    scores = pd.DataFrame({'name':args.label_cols, 'ranksum_score':[scores[col] for col in args.label_cols]})
    scores.to_csv(args.out_file)