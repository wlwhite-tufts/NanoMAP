# Current version of the 2nd step in the old clustering pipeline
# takes the output of an HierarchicalClustering script and finishes the clustering
# probably moving away from this style of clustering, but keep around for legacy purposes

from Bio.Seq import Seq
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import sys
import re
import os
import shutil
import subprocess
from multiprocessing import Pool
from glob import glob
from Levenshtein import distance
from tqdm import tqdm

sys.path.append('/cluster/tufts/cowenlab/wwhite06/packages/VHH-clustering/src/utils/')
from python_utils import *

mmseqs_path = "/cluster/tufts/cowenlab/emosel01/condaenv/vhh/bin/mmseqs" #note that this is Ned's mmseqs because of version issues

# get command line args
args = sys.argv

# Define root_path
root_path = args[1]
if not root_path.endswith('/'):
    root_path += '/'
file_name = args[2]
print(f"Initial Processing of: {file_name}")
# Load data
data = pd.read_csv(root_path+file_name)

##########################
# Section 1:
# Merge families based on CDR3 rules
##########################

# Trim CDR3s
data['junction_aa'] = data['junction_aa'].map(lambda x: x[1:-1])

print(f"There are {len(data['clone_id'].unique())} clonal families initially.")

# calculate distance between families
family_reps = data.loc[data.groupby('clone_id')["VHH_duplicate_count"].idxmax()] #gets the data of the most abundant VHH in each family
dist_matrix = np.array([[CDR3_dist(s1,s2) for s1 in family_reps['junction_aa']] for s2 in tqdm(family_reps['junction_aa'])]).astype(float)
# cluster the clonal families based on the distance rules
clusterer = AgglomerativeClustering(metric='precomputed', linkage='complete', distance_threshold=0.5, n_clusters=None)
family_cluster_labels = clusterer.fit_predict(dist_matrix)

del dist_matrix #clear up space in memory

#go through each group of families that needs to be merged together
for g in np.unique(family_cluster_labels):
    
    #get the group representatives
    group_reps = family_reps[family_cluster_labels == g]
    
    #get the clone id of the most abundant family representative in this group
    group_id = group_reps.loc[group_reps["VHH_duplicate_count"].idxmax(),'clone_id']
    
    print(f'Merging {group_reps.loc[group_reps["VHH_duplicate_count"].idxmax(),"junction_aa"]} ({group_reps.loc[group_reps["VHH_duplicate_count"].idxmax(),"clone_id"]}) with:')
    
    #relabel all families in the group to match the most abundant family
    for f in group_reps['clone_id']:
        
        family = data.loc[data['clone_id']==f,:]
        print(f'\t {family.loc[family["VHH_duplicate_count"].idxmax(),"junction_aa"]} ({family.loc[family["VHH_duplicate_count"].idxmax(),"clone_id"]})')
        
        data.loc[data['clone_id']==f,'clone_id'] = group_id

print(f'{len(data["clone_id"].unique())} clonal families after merging based on CDR3 rules.')
output = root_path + file_name.replace('hclust','merged_by_rules')
data.to_csv(output)

##########################
# Section 2:
# set up and run mmseqs jobs to cluster sequences within each family
##########################

tmp_folders = glob(root_path + 'tmp*')
tmp_name = f'tmp{len(tmp_folders)}'

os.mkdir(root_path + tmp_name)
os.mkdir(root_path + tmp_name + '/fasta')
os.mkdir(root_path + tmp_name + '/mmseqs')

#write fasta file for each family to use as mmseqs input, and collect commands to run
mmseqs_cmds = []
single_vhh_families = []
for f in data['clone_id'].unique():
    
    family = data.loc[data['clone_id']==f,:]
    
    #skip clustering for families with one member
    if len(family) == 1:
        single_vhh_families.append([family['sequence_id'].values[0]]*2)
    
    #make fasta file and mmseqs command for families with >1 member
    else:
        fasta_name = f'{root_path}{tmp_name}/fasta/clone_{f}.fasta'
        with open(fasta_name,'w') as fasta:
            for i,row in family.iterrows():
                fasta.write(f'>{row["sequence_id"]}\n{row["VHH"]}\n')
                
        mmseqs_cmds.append([
            mmseqs_path, 
            'easy-cluster', '-v', '2', '--min-seq-id', '0.90',
            fasta_name,
            f'{root_path}{tmp_name}/mmseqs/clone_{f}',
            f'{root_path}{tmp_name}/tmp_{f}'
            ])
            
single_vhh_families = pd.DataFrame(single_vhh_families,columns=['centroid','sequence_id'])
            
#parallelize all mmseqs commands over the number of cores given to this job
ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
print(f'Splitting {len(mmseqs_cmds)} jobs across {ncpus} cpus.')
pool = Pool(processes=ncpus)
results = pool.map(run_and_clean, mmseqs_cmds)#, chunksize=max(1,int(len(mmseqs_cmds)/(10*n_cpu))))
pool.close()

assert np.all(np.array(results)==0)

#collect mmseqs data
mmseqs_info = pd.concat([pd.read_csv(tsv,names=['centroid','sequence_id'],sep='\t') for tsv in glob(root_path+f'{tmp_name}/mmseqs/*cluster.tsv')],axis=0)
#add back single vhh families
mmseqs_info = pd.concat([mmseqs_info,single_vhh_families],axis=0)

#merge subfamily ids onto main dataframe
print(f'{len(data)} VHHs before merging mmseqs data')
data = data.merge(mmseqs_info,on='sequence_id',how='inner')
print(f'{len(data)} VHHs after merging mmseqs data')

#add family id to subcluster names
data['centroid_id'] = data.apply(lambda r: f"{r['clone_id']}-{r['centroid'].replace('GS-','')}",axis=1)

#write intermediate file
output = root_path + file_name.replace('hclust','subclustered')
data.to_csv(output)

#clean up
shutil.rmtree(f'{root_path}{tmp_name}/mmseqs')
shutil.rmtree(f'{root_path}{tmp_name}/fasta')

##########################
# Section 3:
# use mmseqs info to throw out subfamilies that don't belong
##########################

print(f'{len(data["clone_id"].unique())} clonal families currently.')

#get CDR3 percent homology to most common VHH in the family
for f in data['clone_id'].unique():

    family = data.loc[data['clone_id']==f,:]
    family_rep = family.loc[family['VHH_duplicate_count'].idxmax(),:]
    fam_rep_len = len(family_rep['junction_aa'])
    
    #norma lize dist to the length of the longer sequence, or use 1 when both have no CDR3 to avoid divide by 0 error
    data.loc[family.index,'family_homology_CDR3'] = family['junction_aa'].map(lambda s: 1-(distance(s,family_rep['junction_aa'])/max([fam_rep_len,
                                                                                                                                       len(s),
                                                                                                                                       1])))
    
#determine outliers based on CDR3 homology rules applied to means of subfamilies
for s in data['centroid_id'].unique():
    
    subfamily = data.loc[data['centroid_id']==s,:]
    
    mean_homology = np.mean(subfamily['family_homology_CDR3'])
    cdr3_len = len(subfamily['junction_aa'].values[0])
    
    #require perfect match for small CDR3s
    if cdr3_len <= 6:
        data.loc[subfamily.index,'is_outlier'] = mean_homology < 1
    
    #require >80% match for medium length CDR3s
    if (cdr3_len > 6) and (cdr3_len < 10):
        data.loc[subfamily.index,'is_outlier'] = mean_homology <= 0.8
        
    #require >60% match for long CDR3s
    if cdr3_len >= 10:
        data.loc[subfamily.index,'is_outlier'] = mean_homology <= 0.6
        
#write updated dataframe in place of the one written right after mmseqs
data.to_csv(output)

outliers = data.loc[data['is_outlier'],:].copy()
print(f'Found {len(outliers)} outliers ({round(100*len(outliers)/len(data),1)}% of VHHs)')

##########################
# Section 4:
# recluster (and subcluster) outliers with hierarchical clustering and mmseqs
##########################

if len(outliers):
    
    #remove family and subfamily info before reclustering
    del outliers['clone_id']
    del outliers['centroid']
    del outliers['centroid_id']
    
    outliers.to_csv(f'{root_path}{tmp_name}/outliers.csv')
    
    #run hclust commands in R
    code = subprocess.run(['Rscript', '/cluster/tufts/cowenlab/wwhite06/vhh/scripts/hclust_util.R',
                          f'{root_path}{tmp_name}/outliers.csv', f'{root_path}{tmp_name}/outliers_clustered.csv', '1.25']).returncode
    assert code==0
    
    #get hclust results
    outliers = pd.read_csv(f'{root_path}{tmp_name}/outliers_clustered.csv',index_col=0)
    
    print(f'Found {len(outliers["clone_id"].unique())} clonal families in the outliers.')
    
    os.mkdir(root_path + f'{tmp_name}/fasta')
    os.mkdir(root_path + f'{tmp_name}/mmseqs')

    #write fasta file for each family to use as mmseqs input, and collect commands to run
    mmseqs_cmds = []
    single_vhh_families = []
    for f in outliers['clone_id'].unique():
        
        family = outliers.loc[outliers['clone_id']==f,:]
    
        #skip clustering for families with one member
        if len(family) == 1:
            single_vhh_families.append([family['sequence_id'].values[0]]*2)
        
        #make fasta file and mmseqs command for families with >1 member
        else:
            fasta_name = f'{root_path}{tmp_name}/fasta/clone_{f}.fasta'
            with open(fasta_name,'w') as fasta:
                for i,row in outliers.loc[outliers['clone_id']==f,:].iterrows():
                    fasta.write(f'>{row["sequence_id"]}\n{row["VHH"]}\n')
                    
            mmseqs_cmds.append([
                mmseqs_path, 
                'easy-cluster', '-v', '2', '--min-seq-id', '0.90',
                fasta_name,
                f'{root_path}{tmp_name}/mmseqs/clone_{f}',
                f'{root_path}{tmp_name}/tmp_{f}'
                ])
                
    single_vhh_families = pd.DataFrame(single_vhh_families,columns=['centroid','sequence_id'])
                
    #parallelize all mmseqs commands over the number of cores given to this job
    ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    print(f'Splitting {len(mmseqs_cmds)} jobs across {ncpus} cpus.')
    pool = Pool(processes=ncpus)
    results = pool.map(run_and_clean, mmseqs_cmds)#, chunksize=max(1,int(len(mmseqs_cmds)/(10*n_cpu))))
    pool.close()
    
    assert np.all(np.array(results)==0)
    
    #collect mmseqs data
    mmseqs_info = pd.concat([pd.read_csv(tsv,names=['centroid','sequence_id'],sep='\t') for tsv in glob(root_path+f'{tmp_name}/mmseqs/*cluster.tsv')],axis=0)
    #add back single vhh families
    mmseqs_info = pd.concat([mmseqs_info,single_vhh_families],axis=0)
    
    #merge subfamily ids onto main dataframe
    print(f'{len(outliers)} outlier VHHs before merging mmseqs data')
    outliers = outliers.merge(mmseqs_info,on='sequence_id',how='inner')
    print(f'{len(outliers)} outlier VHHs after merging mmseqs data')
    
    outliers['clone_id'] = outliers['clone_id'].map(lambda x: f'O-{x}')
    outliers['centroid'] = outliers['centroid'].map(lambda x: x.replace('GS','O'))
    outliers['sequence_id'] = outliers['sequence_id'].map(lambda x: x.replace('GS','O'))
    
    #add family id to subcluster names
    outliers['centroid_id'] = outliers.apply(lambda r: f"{r['clone_id']}-{r['centroid'].replace('O-','')}",axis=1)
    
    data = pd.concat([data.loc[~data['is_outlier'],:],outliers],axis=0)
    
    print(f'{len(data["clone_id"].unique())} clonal families including outliers (before final merge).')

#reset index to avoid duplicate indices causing idxmax issues later
data.reset_index(drop=True,inplace=True)

#write intermediate file
output = root_path + file_name.replace('hclust','outliers_subclustered')
data.to_csv(output)

#clean up
if len(outliers):
    shutil.rmtree(f'{root_path}{tmp_name}/mmseqs')
    shutil.rmtree(f'{root_path}{tmp_name}/fasta')

##########################
# Section 5:
# merge families based on CDR3 rules
##########################

# calculate distance between families
family_reps = data.loc[data.groupby('clone_id')["VHH_duplicate_count"].idxmax()] #gets the data of the most abundant VHH in each family
dist_matrix = np.array([[CDR3_dist(s1,s2) for s1 in family_reps['junction_aa']] for s2 in tqdm(family_reps['junction_aa'])]).astype(float)
# cluster the clonal families based on the distance rules
clusterer = AgglomerativeClustering(metric='precomputed', linkage='complete', distance_threshold=0.5, n_clusters=None)
family_cluster_labels = clusterer.fit_predict(dist_matrix)

del dist_matrix #clear up space in memory

#go through each group of families that needs to be merged together
for g in np.unique(family_cluster_labels):
    
    #get the group representatives
    group_reps = family_reps[family_cluster_labels == g]
    
    #get the clone id of the most abundant family representative in this group
    group_id = group_reps.loc[group_reps["VHH_duplicate_count"].idxmax(),'clone_id']
    
    print(f'Merging {group_reps.loc[group_reps["VHH_duplicate_count"].idxmax(),"junction_aa"]} ({group_reps.loc[group_reps["VHH_duplicate_count"].idxmax(),"clone_id"]}) with:')
    
    #relabel all families in the group to match the most abundant family
    for f in group_reps['clone_id']:
        
        family = data.loc[data['clone_id']==f,:]
        print(f'\t {family.loc[family["VHH_duplicate_count"].idxmax(),"junction_aa"]} ({family.loc[family["VHH_duplicate_count"].idxmax(),"clone_id"]})')
        
        data.loc[data['clone_id']==f,'clone_id'] = group_id

print(f'{len(data["clone_id"].unique())} clonal families after merging based on CDR3 rules.')
output = root_path + file_name.replace('hclust','final_clusters')
data.to_csv(output)

#clean up
shutil.rmtree(f'{root_path}{tmp_name}')
