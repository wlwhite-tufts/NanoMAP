import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import homogeneity_score, completeness_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import OPTICS
from scipy.spatial.distance import cdist,pdist
from scipy.special import comb

#get user inputs
parser = argparse.ArgumentParser(prog='Phenotype-based evaluation of clutering quality',
                                 description='Calculate the cluster-size-weighted average within-cluster phenotypic distance.')
#I/O args
parser.add_argument('--in_file',type=str,help='csv file containing sequence and cluster label info')
parser.add_argument('--in_file_header',default=0,type=int,help='number of header rows in input file to skip')
parser.add_argument('--out_file',type=str,help='Name for the output file (can include file path).')
parser.add_argument('--label_columns',type=str,nargs='*',help='Which column(s) contain the cluster labels to be evaluated')
parser.add_argument('--drop_duplicates',type=str,default='VHH',help='Which column to use to drop duplicate entries before scoring. Set to "none" to skip this step.')

#phenotype clustering args
parser.add_argument('--phenotype_cols',type=str,nargs='*', help='Which columns to use for phenotype data when calculating pairwise distances.')
parser.add_argument('--dist_metric',type=str,default='correlation',help="Distance metric to use when clustering by phenotype. Can be any metric allowed by scipy's pdist.")
parser.add_argument('--size_exponent',type=float,nargs='*',default=[1.5],help="Exponent to raise the size of a group to when. If multiple are listed, all are calculated.")
parser.add_argument('--chunk_size',type=int,default=10000,help="Max number of pairwise distances to compute at once.")

def weighted_phenotpye_quality_score(data, id_col, phenotype_cols, metric, exponents, chunk_size):
    
    group_dists = []
    group_sizes = []
    for g, group in data.groupby(id_col):
        
        if len(group) == 1:
            avg_dist = 0
        elif len(group) <= chunk_size:
            avg_dist = np.mean(pdist(group[phenotype_cols].to_numpy(),metric=metric))
        else:
            chunks = [group.iloc[i:min(i+chunk_size, len(group)),:] for i in range(0,len(group),chunk_size)]
            
            tot = 0
            for i,chunk1 in enumerate(chunks):
                for ii,chunk2 in enumerate(chunks[i:]):
                    if i==ii:
                        tot += np.sum(pdist(chunk1[phenotype_cols].to_numpy(),metric=metric))
                    else:
                        tot += np.sum(cdist(chunk1[phenotype_cols].to_numpy(),chunk2[phenotype_cols].to_numpy(),metric=metric))
            avg_dist = tot/comb(len(group),2)
            
        group_dists.append(1-avg_dist)
        group_sizes.append(len(group))
        
    group_dists = np.array(group_dists)
    group_sizes = np.array(group_sizes)
    return [np.sum(group_dists*group_sizes**e) for e in exponents]

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    data = pd.read_csv(args.in_file,header=args.in_file_header)
    
    #remove NaNs
    data = data[~data[args.phenotype_cols].isna().any(axis=1)]
    
    #remove dupilcates if requested
    if args.drop_duplicates != 'none':
        data = data.drop_duplicates(args.drop_duplicates)
    
    if '/' not in args.out_file:
        tmp_dir = './tmp'
    else:
        tmp_dir = '/'.join(args.out_file.split('/')[:-1])+'/tmp'
    
    #remove 0-variace rows because they will mess up distance calculation
    no_var_idx = data[args.phenotype_cols].std(axis=1) < data[args.phenotype_cols][data[args.phenotype_cols]!=0].abs().min().min()/100
    data = data[~no_var_idx]

    #calculate score for each requested label column
    with open(args.out_file,'w') as f:
        f.write('name,'+','.join(['score_'+str(e) for e in args.size_exponent])+'\n')
        for id_col in tqdm(args.label_columns):
            score = weighted_phenotpye_quality_score(data, id_col, args.phenotype_cols, args.dist_metric, args.size_exponent, args.chunk_size)
            score = ','.join([str(s) for s in score])
            f.write(f'{id_col},{score}\n')
            f.flush()