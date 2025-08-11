import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import homogeneity_score, completeness_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import OPTICS
from scipy.spatial.distance import cdist

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
parser.add_argument('--min_clust_frac',type=float,default=0.005,help='Minimum fraction of samples per cluster (also used to determine core samples in DBSCAN).')
parser.add_argument('--xi',type=float,default=0.05,help='Value of xi for use in OPTICS clustering (slope of the reachability plot used to determine cluster boundaries).')

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
    
    # #reserve 0-variace rows because they will mess up distance calculation
    # no_var_idx = data[args.phenotype_cols].std(axis=1) < data[args.phenotype_cols][data[args.phenotype_cols]!=0].abs().min().min()/100
    # reserve = data[no_var_idx]
    # clust = data[~no_var_idx]
    # print(len(clust),len(reserve),flush=True)
    
    # #cluster the samples
    # labels = OPTICS(max_eps=args.dist_cut,eps=args.dist_cut,
    #                 metric=args.dist_metric,
    #                 cluster_method='xi',
    #                 xi=args.xi,
    #                 min_samples=args.min_clust_frac,
    #                 n_jobs=-1).fit_predict(clust[args.phenotype_cols].to_numpy())
                            
    # clust.loc[:,'pheno_label'] = labels
    # #reserved rows get their own cluster label
    # reserve.loc[:,'pheno_label'] = np.max(labels)+1 + (reserve[args.phenotype_cols].mean(axis=1)>0) #this creates one cluster for things that are positive and another for those that are negative
    # data = pd.concat([clust,reserve],axis=0).reset_index()
    
    # print(f'Found {np.sum(labels==-1)} unlabeled samples and {len(data["pheno_label"].unique())-np.any(labels==-1)} clusters.')
    
    # del clust
    # del reserve
    
    # #assign unlabeled points to nearest cluster (if that cluster is close enough)
    # no_label_idx = (data['pheno_label'] == -1).to_numpy()
    
    # #get outlier-to-cluster distances
    # dists = cdist(data.loc[~no_label_idx,args.phenotype_cols].values,
    #               data.loc[no_label_idx,args.phenotype_cols].values,
    #               metric=args.dist_metric)
    # dists = pd.DataFrame(data=dists,index=data.index[~no_label_idx],columns=data.index[no_label_idx])
    # #find closest to each outlier
    # nearest = dists.idxmin(axis=0).to_numpy()
    # close_enough = np.array([dists.loc[nearest[i],col] <= args.dist_cut for i,col in enumerate(dists.columns)])
    # #ignore points that are too far
    # nearest = nearest[close_enough]
    # #assign labels to close enough outliers
    # full_len_close_enough = np.array([close_enough[i] for i in no_label_idx.cumsum()-1])
    # data.loc[no_label_idx&full_len_close_enough,'pheno_label'] = data.loc[nearest,'pheno_label'].tolist()
    
    # #assign new group labels to too far outliers
    # needs_label_idx = data['pheno_label'] == -1
    # N_true_outlier = needs_label_idx.sum()
    # print(f'Assigned unique clusters to {N_true_outlier} unlabeled samples.')
    # data.loc[needs_label_idx,'pheno_label'] = range(data['pheno_label'].max()+1,data['pheno_label'].max()+1+N_true_outlier)
    
    data.loc[:,args.phenotype_cols] = np.sign(data.loc[:,args.phenotype_cols])
    data['pheno_label'] = np.nan
    for i,(_,group) in enumerate(data.groupby(args.phenotype_cols)):
        data.loc[group.index,'pheno_label'] = i
    
    print(f'Found {len(data["pheno_label"].unique())} phenoytpe clusters (in {len(data)} datapoints). {(data["pheno_label"]==-1).sum()}')

    #calculate score for each requested label column
    with open(args.out_file,'w') as f:
        f.write('name,homogeneity,completeness,ARI\n')
        for id_col in tqdm(args.label_columns):
            hom_score = homogeneity_score(data['pheno_label'].values,data[id_col].astype(str).values)
            com_score = completeness_score(data['pheno_label'].values,data[id_col].astype(str).values)
            ari_score = adjusted_rand_score(data['pheno_label'].values,data[id_col].astype(str).values)
            # sil_score = silhouette_score(data)
            f.write(f'{id_col},{hom_score},{com_score},{ari_score}\n')
            f.flush()