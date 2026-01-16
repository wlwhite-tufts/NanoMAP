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
                                     description='Score clusterings based on the phenotypic consistency within clusters.')
    #I/O args
    parser.add_argument('--in_file',type=str,help='csv file containing sequence and cluster label info')
    parser.add_argument('--out_file',type=str,help='Name for the output file (can include file path).')
    parser.add_argument('--label_cols',type=str,nargs='*',help='Which column(s) contain the cluster labels to be evaluated')
    parser.add_argument('--in_file_header',default=0,type=int,help='number of header rows in input file to skip')
    parser.add_argument('--drop_duplicates',type=str,default='VHH',help='Which column to use to drop duplicate entries before scoring. Set to "none" to skip this step.')
    
    #phenotype args
    parser.add_argument('--phenotype_cols',type=str,nargs='*', help='Which columns to use for phenotype data when calculating pairwise distances.')
    parser.add_argument('--dist_metric',type=str,default='correlation',help="Distance metric to use when clustering by phenotype. Can be any metric allowed by scipy's pdist.")

    return parser.parse_args()
    
def ranksum_score_single(dists_within, dists_between):
    '''
    Calculate the Z-statistic from a Mann-Whitney ranksum test comparing the toe provided lists of values.
    
    Parameters
    ----------
    dists_within: array
        numpy array containing the distances between a single sequence and the members of its cluster
    dists_between: array
        numpy array containing the distances between a single sequence and the members of all other clusters
        
    Returns
    -------
    score: float
        the z-statistic derived from comparing the within- and between-cluster distances
    '''
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
        
    score = ranksums(dists_within,dists_between,
                    alternative='less',
                    nan_policy='omit').statistic
    
    return score
    
def ranksum_score_multilabel(idx, N, N_labels, N_pheno, dist_metric, rand):
    '''
    For a given sequence, calculate the "ranksum score" using each of the given clusterings.
    The cluster labels are provided in a shared memory object to allow more efficient parallel processing.
    
    Parameters
    ----------
    idx: int
        the row index in the shared memory objects corresponding to the sequence of interest
    N: int
        the number of sequences in the dataset (needed to properly load shared memory objects)
    N_labels: int
        the number of different clusterings to be scored (needed to properly load shared memory objects)
    N_pheno: int
        the number of phenotypic datapoints associated with each sequence (needed to properly load shared memory objects)
    dist_metric: str
        the distance metric to use when calculating the phenotypic distances between sequences
    rand: int
        the randomly generated integer used in naming the shared memory objects (used to avoid collisions when running multiple jobs at once)
    
    Returns
    -------
    scores: list
        the scores for this sequence, one for each clustering present in the shared memory object
        scores are listed in the same order that clustering are provided in the columns of the shred memory object
    '''
    #load in cluster label shared memory object
    label_shm = SharedMemory(name=f'labels_{rand}')
    labels = np.ndarray((N,N_labels), dtype=np.double, buffer=label_shm.buf)
    
    #load in phenotypic data shared memory object
    pheno_shm = SharedMemory(name=f'pheno_{rand}')
    pheno = np.ndarray((N,N_pheno), dtype=np.double, buffer=pheno_shm.buf)
    
    #get the distances between the sequence of interest and all others
    pheno_row = pheno[idx,:]
    dists = cdist(np.expand_dims(pheno_row,0),pheno,metric=dist_metric).squeeze()
    
    #get a score for this sequence for each clustering provided     
    not_nan = ~np.isnan(dists)
    scores = []
    for c in range(labels.shape[1]):
        #based on the current clustering, separate distances to seqs that match vs do not match the cluster of the sequence of interest
        in_clust = labels[:,c]==labels[idx,c]
        dists_within = dists[in_clust&not_nan]
        dists_between = dists[(~in_clust)&not_nan]
        
        #calculatea and save the score
        scores.append(ranksum_score_single(dists_within, dists_between))
    
    #close shared memory
    label_shm.close()
    pheno_shm.close()
    
    print(idx,flush=True)
        
    return scores
    
def ranksum_score_all(data, label_cols, phenotype_cols, dist_metric):
    '''
    Score all sequences and clusterings in a dataset using the "ranksum score", and average over the sequences.
    
    Parameters
    ----------
    data: DataFrame
        pandas DataFrame containing the cluster labels and phenotypic data
        must include columns matching the column names in label_cols and phenotype_cols
    label_cols: list
        list of column names in data that contain cluster labels
        one column per cluster that should be scored
    phenotype_cols: list
        list of column names in data that contain binding information
        will be treated as vectors with values in the order provided when calculating "phenotypic distance"
    dist_metric: str
        the distance metric to use when calculating the phenotypic distances between sequences
        
    Returns
    -------
    scores: dict
        the average score for each clustering in label_cols (averaged over all sequences in the dataset)
        keys are column names and values are the corresponding average score
    '''
    #get dataset dimensions
    N = len(data)
    N_labels = len(label_cols)
    N_pheno = len(phenotype_cols)
    
    #generate random label for the dataset
    rand = np.random.randint(0,high=1e6)
    
    #put cluster label data into shared memory
    label_mem = SharedMemory(name=f'labels_{rand}', create=True, size=8*N*N_labels)
    labels = np.ndarray((N,N_labels), dtype=np.double, buffer=label_mem.buf)
    labels[:] = data[label_cols].to_numpy()[:]
    
    #put phenotypic data into shared memory
    pheno_mem = SharedMemory(name=f'pheno_{rand}', create=True, size=8*N*N_pheno)
    pheno = np.ndarray((N,N_pheno), dtype=np.double, buffer=pheno_mem.buf)
    pheno[:] = data[phenotype_cols].to_numpy()[:]
    
    #connect to paralle pool
    ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    pool = Pool(processes=ncpus)
    print(f'Connected to pool with {ncpus} cpus.',flush=True)
    
    #score each sequence in parallel, then collect and average scores accross sequences
    scores = np.array(pool.starmap(ranksum_score_multilabel, ([idx, N, N_labels, N_pheno, dist_metric, rand] for idx in range(N))))
    scores = np.nanmean(scores,axis=0)
    scores = {col:scores[c] for c,col in enumerate(label_cols)}
    
    #close parallel pool and shared memory
    pool.close()
    label_mem.close()
    label_mem.unlink()
    pheno_mem.close()
    pheno_mem.unlink()
    
    return scores
    
def fix_label_type(obj):
    '''
    Convert sequence IDs starting with "GS-" to integers so that converting to numpy shared memory object is possible
    
    Parameters
    ----------
    obj: str or int
        the object to convert to an integer, if needed
        must either be an integer already, or be a string starting with "GS-" followed by an integer
        
    Returns
    -------
    obj: int
        the integer verison of the object
    '''
    if type(obj)==str and obj.startswith('GS-'):
        return int(obj.split('-')[1])
    return obj
    
if __name__ == '__main__':
    
    args = get_args()
    
    #read in data file, dropping duplicates if requested
    data = pd.read_csv(args.in_file,header=args.in_file_header)
    if args.drop_duplicates.lower() != 'none':
        data = data.drop_duplicates(args.drop_duplicates)
    
    #convert sequence IDs labels to integers
    for col in args.label_cols:
        data[col] = data[col].map(fix_label_type) 
    
    #calculate scores, convert to DataFrame, and save
    scores = ranksum_score_all(data, args.label_cols, args.phenotype_cols, args.dist_metric)
    scores = pd.DataFrame({'name':args.label_cols, 'ranksum_score':[scores[col] for col in args.label_cols]})
    scores.to_csv(args.out_file)