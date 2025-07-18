import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import homogeneity_score
from scipy.cluster.hierarchy import fclusterdata

#get user inputs
parser = argparse.ArgumentParser(prog='Phenotype-based evaluation of clutering quality',
                                 description='Calculate the silhouette score based on correlation distance of user-specified phenotype data for a given (set of) clustering(s).')
#I/O args
parser.add_argument('--in_file',type=str,help='csv file containing sequence and cluster label info')
parser.add_argument('--in_file_header',default=0,type=int,help='number of header rows in input file to skip')
parser.add_argument('--out_file',type=str,help='Name for the output file (can include file path).')
parser.add_argument('--label_columns',type=str,nargs='*',help='Which column(s) contain the cluster labels to be evaluated')
parser.add_argument('--drop_duplicates',type=str,default='VHH',help='Which column to use to drop duplicate entries before scoring. Set to "none" to skip this step.')

#phenotype clustering args
parser.add_argument('--phenotype_cols',type=str,nargs='*', help='Which columns to use for phenotype data when calculating pairwise distances.')
parser.add_argument('--dist_metric',type=str,default='correlation',help="Distance metric to use when clustering by phenotype. Can be any metric allowed by scipy's pdist.")
parser.add_argument('--dist_cut',type=float,default=0.5,help='Distance cutoff used to define phenotypic clusters.')
parser.add_argument('--linkage',type=str,default='average',help='Linkage method for agglomerative clustering on phenotype.')

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
    
    #reserve 0-variace rows because they will mess up distance calculation
    no_var_idx = data[args.phenotype_cols].std(axis=1) < data[args.phenotype_cols][data[args.phenotype_cols]!=0].abs().min().min()/100
    reserve = data[no_var_idx]
    clust = data[~no_var_idx]
    print(len(clust),len(reserve),flush=True)
    
    #cluster the samples
    labels = fclusterdata(clust[args.phenotype_cols].values,
                            metric=args.dist_metric,
                            method=args.linkage,
                            t=args.dist_cut,
                            criterion='distance')
                            
    clust['pheno_label'] = labels
    #reserved rows get their own cluster label
    reserve['pheno_label'] = np.max(labels)+1 + (reserve[args.phenotype_cols].mean(axis=1)>0) #this creates one cluster for things that are positive and another for those that are negative
    data = pd.concat([clust,reserve],axis=0)
    
    print(f'Found {len(np.unique(labels))} phenoytpe clusters (in {len(data)} datapoints).')

    #calculate score for each requested label column
    with open(args.out_file,'w') as f:
        f.write('name,score\n')
        for id_col in tqdm(args.label_columns):
            score = homogeneity_score(data['pheno_label'].values,data[id_col].astype(str).values)
            f.write(f'{id_col},{score}\n')
            f.flush()