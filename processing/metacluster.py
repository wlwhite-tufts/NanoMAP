import numpy as np
import pandas as pd
import sys
import os
import shutil
import subprocess
from sklearn.cluster import AgglomerativeClustering
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm
import argparse
import itertools
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from Levenshtein import distance
from functools import partial
from itertools import repeat

#get main dir for repo
repo_path = os.path.dirname(os.path.abspath(__file__))
repo_path = '/'.join(repo_path.split('/')[:-1])
sys.path.append(repo_path)
from utils import *
from analysis.fast_silhouette_score import weighted_ANARCI_silhouette_fast

def get_args():
    #get user inputs
    parser = argparse.ArgumentParser(prog='Cluster Based on ANARCI alignment',
                                     description='Collect sequences from input folder and cluster them using an agglomerative clustering method.')
    #I/O args
    parser.add_argument('--in_dir',type=str,help='The directory that contains the assembled sequences. Looks for *db-pass.tsv, or */*db-pass.tsv')
    parser.add_argument('--out_file',type=str,help='Name for the output file (can include file path).')
    parser.add_argument('--keep_intermediates',action='store_true',help='Set this flag to save the cluster IDs for each clustering leading to the metaclustering.')
    
    #translation args
    parser.add_argument('--start_seq',type=str,default='LVQPGGSLRLSC',help='Amino acid sequence to scan for when finding starting point of translation.')
    parser.add_argument('--end_seq',type=str,default='WGQGTQVTVSS',help='Amino acid sequence to scan for when finding ending point of translation.')
    parser.add_argument('--min_start_match',type=int,default=5,help='Minimum number of amino acids matching start_seq to be considered a real sequence.')
    parser.add_argument('--min_end_match',type=int,default=5,help='Minimum number of amino acids matching end_seq to be considered a real sequence.')
    parser.add_argument('--max_start_scan',type=int,default=40,help='Number of N-term amino acid positions to scan when looking for start_seq.')
    parser.add_argument('--max_end_scan',type=int,default=40,help='Number of C-term amino acid positions to scan when looking for end_seq.')
    parser.add_argument('--min_count',type=int,default=2,help='The minimum number of counts for an amino acid sequence for it to be retained for clustering')
    
    #clustering args
    parser.add_argument('--max_subgroup_size',type=int,default=25000,help='Maximum number of VHHs in a SCOPer (sub)group before triggering a recursive loose mmseqs sub-clustering.')
    parser.add_argument('--recursive_min_id_start',type=float,nargs=2,default=[0.5,0.8],help='Inclusive bounds for range of starting values for minimum sequence identity allowed for recursive loose mmseqs clustering.')
    parser.add_argument('--recursive_min_id_start_N',type=int,default=5,help='Number of values to test between the recursive_min_id_start bounds.')
    parser.add_argument('--recursive_min_id_step',type=float,default=0.035,help='Step size value for minimum sequence identity incerment in each recursive layer.')
    
    parser.add_argument('--recursive_method',type=str,nargs='*',default=['setcover'],help='Method to use for clustering in the recursive mmseqs clustering (setcover or conncomp).')
    
    parser.add_argument('--CDR3_weight',type=float,nargs=2,default=[3,3], help='Inclusive bounds for the weight for CDR3 in weighted distance calculations. Uses a weight of 1 for CDR1 and CDR2.')
    parser.add_argument('--CDR3_weight_N',type=int,default=1,help='Number of values to test between the CDR3_weight bounds.')
    
    parser.add_argument('--max_len_diff',type=int,nargs=2,default=[3,3], help='Inclusive bounds for the range of values of maximum allowed difference in CDR length.')
    parser.add_argument('--max_len_diff_N',type=int,default=1,help='Number of values to test between the max_len_diff bounds.')
    
    parser.add_argument('--linkage',type=str,default=['single','average','complete'],nargs='+',help='The linkage methods to try in weighted hierarchical clustering step (can be single, complete, or average).')
    
    parser.add_argument('--weighted_min_id',type=float,nargs=2,default=[0.4,0.8],help='Inclusinve bounds for the range of distance cutoff values to use in the weighted hierarchical clustering step.')
    parser.add_argument('--weighted_min_id_N',type=int,default=5,help='Number of values to test between the weighted_min_id bounds.')
    
    parser.add_argument('--final_merge_method',type=str,default=['complete'],nargs='+',help='The linkage method(s) to use in the final CDR-based merge.')
    
    parser.add_argument('--final_min_id',type=float,nargs=2,default=[0.75,0.95],help='Inclusinve bounds for the range of distance cutoff values to use in the final merge step.')
    parser.add_argument('--final_min_id_N',type=int,default=3,help='Number of values to test between the final_min_id bounds.')
    
    parser.add_argument('--dist_batch_size',type=int,default=1000,help='When calculating all pairwise distances between sequences in a set, break the calculation into sets of dist_batch_size (so that dist_batch_size**2 distances are calculated per batch).')
    
    #scoring args
    parser.add_argument('--score_fraction',type=float,default=0.5,help='Keep the top score_fraction of individual clusterings. Only these clusterings are used in metaclustering.')
    parser.add_argument('--sample_N_pairs',type=int,default=10000,help='When calculating intra-family distances, calcualte weighted distances for a random sampling of sample_N_pairs pairs of sequences within the cluster.')
    
    #metaclusterign args
    parser.add_argument('--meta_min_id',type=float,default=[0.25],nargs='*',help='Minimum fraction of agreeing clusterings to group two VHHs. If given multiple values will try all. Clustering distance threshhold = 1 - meta_min_id')
    parser.add_argument('--meta_method',type=str,default=['single'],nargs='*',help='Linkage method to use for meta clustering (can be single, average, or complete). If given multiple values will try all.')
    
    return parser.parse_args()

def collect_seqs(files):
    '''
    Concatenate data from all tsv files in the provided list.
    No deduplication is performed.
    
    Parameters
    ----------
    files: list
        the list of file names to concatenate
        these should be the *_db-pass.tsv files generated by assemble_and_filter_reads.py
        
    Returns
    -------
    data: DataFrame
        pandas DataFrame with all requested files concatenated.
    '''
        
    print('Reading the following files:')
    print('\n'.join([f.split('/')[-1] for f in files]))
    data = pd.concat([pd.read_csv(tsv,sep='\t') for tsv in tqdm(files)],axis=0)
    #drop less useful info
    data = data[["sequence", "locus", "junction", "junction_aa", "v_call", "d_call", "j_call", "duplicate_count"]]
    
    print(f'{len(data)} rows in global set after loading tsvs.', flush=True)
    return data

def annotate_and_filter_seqs(args, data, pool, ncpus, agg_funcs):
    '''
    Take an input DataFrame, remove bad sequeces from it, and use ANARCI to determine CDR sequences.
    Removes the following types of sequences:
        Sequences with internal stop codons
        Sequences with N or C termini that do not resemble VHHs
        Rare sequences
    Adds the following info columns for each sequence
        sequence_id: a unique string identifying each sequence
        VHH: the translated sequence
        start_match: the number of amino acids at the beginning of the translation that match args.start_seq
        end_match: the number of amino acids at the end of the translation that match args.end_seq
        start_i: the position in the DNA sequence where the translation started
        end_i: the position in the DNA sequence where the translation ended
        start_f: which frame the rtanslation starts in
        CDR1: the CDR1 amino acid sequence
        CDR2: the CDR2 amino acid sequence
        CDR3: the CDR3 amino acid sequence
        anarci_parse: a tuple of numpy arrays with annotations from ANARCI
            anarci_parse[0] is a character array with the amino acid sequence
            anarci_parse[1] is an integer array such that anarci_parse[1][i] is th IMGT position of anarci_parse[0][i]
        ANARCI_success: bool indicating if the ANARCI parsing was successful
    
    Parameters
    ----------
    args: parsed arguments
        The arguments passed as input into the main script - used for start and end point determination and filtering
        Must contain the following fields:
            start_seq: the amino acid sequence expected at the N-term of all VHHs
            end_seq: the amino acid sequence expected at the C-term of all VHHs
            max_start_scan: the maximal number of bases to scan into the 5' end of the DNA sequence when looking for a translation match to start_seq
            max_end_scan: the maximal number of bases to scan into the 3' end of the DNA sequence when looking for a translation match to end_seq
            min_start_match: the minimum number of amino acids matching start_seq for the translation to be considered successful
            min_end_match: the minimum number of amino acids matching end_seq for the translation to be considered successful
            min_count: the minimum number of read counts associated with an amino acid sequence for it to be retained in the dataset
    
    data: DataFramw
        pandas DataFrame containing the sequences to be annotated and filtered
        Must have the following columns:
            sequence: the DNA sequence of the VHH
            duplicate_count: the number of read counts associated with each DNA sequence
            
    pool: Pool
        a parallel pool object to use for parallel processing of translation and ANARCI annotation
        
    ncpus: int
        the number of cpus that pool has access to
        
    agg_funcs: dict
        dictionary mapping columns in data to functions that will be used to aggregate the data in those columns when aggregating by amino acid sequence
        
    Returns
    -------
    data: DataFrame
        The modified version of data including new annotation columns, and with the bad sequences filtered out
        
    vhh: DataFrame
        The same information as in data, but aggregated by amino acid sequence instead of DNA sequence
    '''
    # Aggregate duplicate DNA sequences
    data = data.groupby('sequence', sort=False).aggregate(func=agg_funcs).reset_index(drop=True)
    print(f'{len(data)} unique DNA sequences.', flush=True)

    # Parallel translation of sequences
    # Dynamically choose a good chunksize
    chunksize = max(len(data) // (ncpus * 4), 1000)
    
    # Efficient parallel translation
    results = pool.imap_unordered(
        translate_wrapper,
        zip(
            data['sequence'],
            repeat(args.start_seq),
            repeat(args.end_seq),
            repeat(args.max_start_scan),
            repeat(args.max_end_scan)
        ),
        chunksize=chunksize
    )
    
    # Collect results into DataFrame
    seq_info = pd.DataFrame(list(results), columns=['sequence','VHH','start_match','start_i','start_f','end_match','end_i'])

    # Merge translation results into original dataframe
    data = data.merge(seq_info, on='sequence', how='inner')

    # Filter out sequences with stop codons, poor start/end matches, or invalid translations
    data = data[
        data['VHH'].notna() &
        (~data['VHH'].str.contains(r'\*', regex=True)) &
        (data['start_match'] >= args.min_start_match) &
        (data['end_match'] >= args.min_end_match)
    ]
    print(f'{len(data)} sequences passed translation and motif filtering.', flush=True)

    # Compute VHH-level duplicate counts and merge
    vhh_counts = data.groupby('VHH',sort=False)['duplicate_count'].sum().rename('VHH_duplicate_count').reset_index()
    data = data.merge(vhh_counts, on='VHH')

    # Drop singleton VHHs
    data = data[data['VHH_duplicate_count'] >= args.min_count]
    print(f'{len(data)} sequences, encoding {data["VHH"].nunique()} unique VHHs after removing rare amino acid sequences (<{args.min_count}).', flush=True)

    # Reset index and assign sequence IDs
    data = data.reset_index(drop=True)
    data['sequence_id'] = data.index.map(lambda i: 'GS-' + str(i))

    # Drop duplicate full-length VHHs (keep most abundant DNA encoding)
    data = data.sort_values(by='duplicate_count', ascending=False)
    vhh = data.drop_duplicates('VHH', keep='first')

    print(f'{len(vhh)} unique VHHs after deduplication.', flush=True)

    # Get ANARCI annotations
    print(f'Getting ANARCI parses.', flush=True)
    anarci_info = pd.DataFrame(pool.map(get_anarci_alignment, vhh['VHH'], chunksize=1000),
                               columns=['VHH', 'anarci_parse', 'CDR1', 'CDR2', 'CDR3', 'ANARCI_success'])
    vhh = vhh.merge(anarci_info, on='VHH', how='left')
    print(f'{len(vhh)} rows after merging with ANARCI data.', flush=True)

    return data, vhh

def metacluster(args,vhh,out_dir,out_fname,pool,ncpus):
    '''
    Main function for calculating the meta-clustering for a given dataset.
    Returns a modified version of the input DataFrame with column(s) for containing cluster labels for the requested clustering(s).
    Initial clusterings will be provided in columns with names starting with clone_id_
    Meta-clustering will be provided in columns with names starting with meta_clone_id_
    
    Parameters
    ----------
    args: parsed arguments
        The arguments passed as input into the main script - used for start and end point determination and filtering
        Must contain the following fields:
            recursive_method: clustering method for mmseqs to use in recursive splitting step (setcover or conncomp)
            recursive_min_id_start: list containing start and end values defining the range of values to use as a starting point for recursive splitting
            recursive_min_id_start_N: the number of values within the range defined by recursive_min_id_start to use 
            recursive_min_id_step: the increment to increase the recursive_min_id by in each recursive layer
            max_subgroup_size: the maximum number of sequences in a subgroup produced by the recursove clustering step
            CDR3_weight: list containing start and end values defining the range of values to use as the weight of CDR3 mismatches relative to CDR1 and 2
            CDR3_weight_N: the number of values within the range defined by CDR3_weight to use
            max_len_diff: list containing start and end values defining the range of values to use as the maximum allowed difference in CDR lenghts (applied to each CDR separately)
            max_len_diff_N: the number of values within the range defined by max_len_diff to use
            linkage: list of the clustering methods to use during the fine-grained clustering step
            weighted_min_id: list containing start and end values defining the range of values to use as the minimum identity required for co-clustering
            weighted_min_id_N: the number of values within the range defined by weighted_min_id to use
            final_merge_method: list of the clustering methods to use during the merging step
            final_min_id: list containing start and end values defining the range of values to use as the minimum identity for cluster merging
            final_min_id_N: the number of values within the range defined by final_min_id to use
            dist_batch_size: the maximum number of sequences to include in each side of pairwise distance batches (dist_batch_size**2 distances are calculated in each batch)
            score_fraction: the fraction of initial clusterings to keep for meta-clustering (those with the highest fast silhouette scores are kept)
            sample_N_pairs: number of pairs to sample between and within clusters when calculating the fast silhouette score
            meta_min_id: list containing all values to use as minmum fraction of matching labels for meta-clustering
            meta_method: list of clustering methods to use in for meta-clustering
            keep_intermediates: boolean flag indicating if the initial clustering columns should be retained in the output
    
    vhh: DataFrame
        pandas DataFrame with the sequences to be clustered
        Must contain the following columns:
            VHH: the amino acid sequence of the VHH
            sequence_id: a unique idntifier for the VHH
            VHH_duplicate_count: the number of read counts assocaited with the VHH amino acid sequence
            CDR1: the CDR1 amino acid sequence
            CDR2: the CDR2 amino acid sequence
            CDR3: the CDR3 amino acid sequence
            
    out_dir: str
        the file path to the directory where temporary files will be written
        
    out_name: str
        the name of the desired output file
        used for naming temporary files to avoid conflict with other temporary files being written at the same time
        
    pool: Pool
        a parallel pool object to use for parallel processing of translation and ANARCI annotation
        
    ncpus: int
        the number of cpus that pool has access to
        
    Returns
    -------
    vhh: DataFrame
        modified version of the input DataFrame with clustering info added
        
    id_cols: list
        list of the column names that contain initial clustering labels
        
    meta_id_cols: list
        list of the column names that contain meta-clustering labels
    '''
    #############
    ## get parameter combos
    #############
    recursive_min_id_vals = itertools.product(args.recursive_method,
                                              np.linspace(*args.recursive_min_id_start,args.recursive_min_id_start_N))
    recursive_min_id_vals = list(recursive_min_id_vals)
    
    cdr3_dist_combos = itertools.product(np.linspace(*args.CDR3_weight,args.CDR3_weight_N),
                                         np.linspace(*args.max_len_diff,args.max_len_diff_N))
    cdr3_dist_combos = list(cdr3_dist_combos)
                                         
    weighted_clustering_combos = itertools.product(args.linkage,
                                                   np.linspace(*args.weighted_min_id,args.weighted_min_id_N))
    weighted_clustering_combos = list(weighted_clustering_combos)
    
    final_merge_combos = itertools.product(args.final_merge_method,
                                           np.linspace(*args.final_min_id,args.final_min_id_N))
    final_merge_combos = list(final_merge_combos)

    #############
    ## group sequences based on V and J identity
    #############
    
    tmp_dir = f'{out_dir}tmp_{out_fname.replace(".csv","")}'
    if os.path.exists(tmp_dir):
        #remove stuff from previous run if it exists
        shutil.rmtree(tmp_dir)
    #make the folder fresh
    os.mkdir(tmp_dir)
    
    print(f'Getting SCOPer groups.', flush=True)
    
    #get basic V/J clustering from Immcantation (scoper::hierarchicalClones)
    #make fake column to spoof CDR length/clustering
    vhh['fake_junction'] = 'A'
    #run hclust in R
    vhh.to_csv(f'{out_dir}tmp_input_{out_fname}')
    code = subprocess.run(['Rscript', '/cluster/tufts/cowenlab/wwhite06/vhh/scripts/hclust_V-J-len_only_util.R',
                          f'{out_dir}tmp_input_{out_fname}', f'{out_dir}tmp_output_{out_fname}', 'fake_junction']).returncode
    assert code==0
    #get hclust data
    vhh = pd.read_csv(f'{out_dir}tmp_output_{out_fname}',na_filter=False)
    #clean up tmp files
    os.remove(f'{out_dir}tmp_input_{out_fname}')
    os.remove(f'{out_dir}tmp_output_{out_fname}')
    
    #delete fake column
    del vhh['fake_junction']
    
    #############
    ## cluster sequences based full length AA sequence and weighting scheme
    #       repeat this for all possible parameter combos given by user
    #############
    
    #cluster within SCOPer groups
    min_id = defaultdict(int)
    for g,scoper_group in vhh.groupby(by='scoper_group',sort=False):
        
        #try all recursive_min_id values for this SCOPer group
        for rec_idx,rec_params in enumerate(recursive_min_id_vals):
            rec_min_id = rec_params[1]
            rec_method = rec_params[0]
        
            #recursively split into smaller subgroups with mmseqs
            scoper_group, max_min_id = recursive_mmseqs_split(scoper_group, args.max_subgroup_size, rec_min_id, args.recursive_min_id_step, rec_method, tmp_dir, ncpus)
            
            #cluster by weighted CDR distance within each broad mmseqs cluster
            for mm_id,mmseqs_group in tqdm(scoper_group.groupby('mmseqs_id',sort=False)):
                
                #try all CDR3 weight values
                for cdr_idx,dist_params in enumerate(cdr3_dist_combos):
                    
                    # Map row positions for iloc
                    row_pos = vhh.index.get_indexer(mmseqs_group.index)
                
                    #skip this group if it has only one member
                    if len(mmseqs_group) == 1:
                        
                        #save ID number for all clustering param combos
                        for clust_idx,_ in enumerate(weighted_clustering_combos):
                            
                            #save ID number for all clustering param combos
                            clust_col = f'clone_id_{rec_idx}_{cdr_idx}_{clust_idx}'

                            # Initialize column if it doesn't exist
                            if clust_col not in vhh.columns:
                                vhh[clust_col] = np.nan

                            #record cluster label and dist to furthest within cluster
                            vhh.iloc[row_pos, vhh.columns.get_loc(clust_col)] = min_id[(rec_idx,cdr_idx,clust_idx)] #use min_id as usual

                    #actually do clustering for groups with >1 member
                    else:
                        #get dist matrix
                        CDR_list = [row[['CDR1','CDR2','CDR3']].tolist() for _,row in mmseqs_group.iterrows()]
                        dists = batched_pairwise_weighted_anarci_dist(CDR_list, args.dist_batch_size, [1,1,dist_params[0]], [dist_params[1]]*3, pool)

                        #try all clustering param combos
                        for clust_idx,clust_params in enumerate(weighted_clustering_combos):
                            clusterer = AgglomerativeClustering(metric='precomputed',
                                                                linkage=clust_params[0],
                                                                distance_threshold=1-clust_params[1],
                                                                n_clusters=None)
                            labels = clusterer.fit_predict(dists) + min_id[(rec_idx,cdr_idx,clust_idx)]

                            #merge labels onto group info
                            clust_col = f'clone_id_{rec_idx}_{cdr_idx}_{clust_idx}'
                            # dist_col = f'farthest_{rec_idx}_{cdr_idx}_{clust_idx}'
                            
                            # Initialize column if it doesn't exist
                            if clust_col not in vhh.columns:
                                vhh[clust_col] = np.nan
                                # vhh[dist_col] = np.nan
                            
                            # Assign using iloc for speed
                            vhh.iloc[row_pos, vhh.columns.get_loc(clust_col)] = labels
                    
                    #update min_id for all clustering variations
                    for clust_idx,_ in enumerate(weighted_clustering_combos):
                        min_id[(rec_idx,cdr_idx,clust_idx)] = vhh[f'clone_id_{rec_idx}_{cdr_idx}_{clust_idx}'].max() + 1
                
            #delete mmseqs_id column so it can be re-used in the next mmseqs recursive split
            del scoper_group['mmseqs_id']
            
            #defragment the dataframe
            vhh = vhh.copy()
                
    print(f'Finished primary individual clusterings, starting merges.', flush=True)
    
    vhh.to_csv(f'{tmp_dir}/intermediate.csv')
    
    #############
    ## merge very similar clusters (using CDR distance)
    #############
    
    id_cols = [col for col in vhh.columns if 'clone_id' in col]
    
    #sort vhhs by duplicate count so drop duplicates will keep the most abundant ones
    vhh = vhh.sort_values(by='VHH_duplicate_count',ascending=False)
    
    #gather all representatives of all clusters
    rep_seqs_master = pd.DataFrame(columns=vhh.columns)
    for id_col in id_cols:
        rep_seqs_master = pd.concat([rep_seqs_master,vhh.drop_duplicates(id_col,keep='first')]).drop_duplicates('sequence_id')
    rep_seqs_master.reset_index(inplace=True,drop=True)

    #split into rough groups with mmseqs
    rep_seqs_master = run_mmseqs_clust(rep_seqs_master,
                                       args.final_min_id[0], #use the lowest merge value, to be safe (full seq id used here should be higher than CDR dist anyway)
                                       'conncomp', tmp_dir, ncpus)
    
    rep_groups = rep_seqs_master.groupby('mmseqs_id',sort=False)
    rep_sizes = rep_groups.size()
    print(f'Split {len(rep_seqs_master)} cluster representatives into {len(rep_sizes)} groups, with max size = {rep_sizes.max()}.',flush=True)
                                
    #calculate all pairwise distances within mmseqs groups
    dist_dict = {}
    for mm_id,mmseqs_group in tqdm(rep_groups):
        if len(mmseqs_group) > 1:
            CDR_list = [row[['CDR1','CDR2','CDR3']].tolist() for _,row in mmseqs_group.iterrows()]
            dists = batched_pairwise_weighted_anarci_dist(CDR_list, args.dist_batch_size, [1,1,np.mean(args.CDR3_weight)], [np.mean(args.max_len_diff)]*3, pool)
            
            #convert to DataFrame
            dists = pd.DataFrame(dists,index=mmseqs_group['sequence_id'],columns=mmseqs_group['sequence_id'])
            dist_dict[mm_id] = dists
    
    #for each of the above clusterings ... try all requested merging values
    for id_col in tqdm(id_cols):
        rep_seqs = rep_seqs_master.drop_duplicates(id_col,keep='first').copy()
        
        min_id = defaultdict(int)
        #merge within each mmseqs group
        for mm_id,mmseqs_group in rep_seqs.groupby('mmseqs_id',sort=False):
            
            #skip clustering if group size is 1
            if len(mmseqs_group) == 1:
                for merge_idx,(merge_method, merge_min_id) in enumerate(final_merge_combos):
                    rep_seqs.loc[mmseqs_group.index,f'{id_col}_{merge_idx}'] = min_id[(merge_method, merge_min_id)]
                    min_id[(merge_method, merge_min_id)] += 1
            else:
                #look up reelvant dists in pre-calculated dict
                dists = dist_dict[mm_id].loc[mmseqs_group['sequence_id'],mmseqs_group['sequence_id']]
        
                #try each requested merge min_id and method
                for merge_idx,(merge_method, merge_min_id) in enumerate(final_merge_combos):
                    
                    #get labels
                    clusterer = AgglomerativeClustering(metric='precomputed',
                                                        linkage=merge_method,
                                                        distance_threshold=1-merge_min_id,
                                                        n_clusters=None)
                    #insert labels
                    labels = clusterer.fit_predict(dists.to_numpy(copy=True)) + min_id[(merge_method, merge_min_id)]
                    rep_seqs.loc[mmseqs_group.index,f'{id_col}_{merge_idx}'] = labels
                    
                    #update min_id for next group
                    min_id[(merge_method, merge_min_id)] = rep_seqs[f'{id_col}_{merge_idx}'].max() + 1
        
        #merge on final cluster info
        vhh = vhh.merge(rep_seqs[[id_col]+[f'{id_col}_{merge_idx}' for merge_idx in range(len(final_merge_combos))]],on=id_col,how='left')
                
        #delete original column since it is no longer needed
        del vhh[id_col]

        #defragment by copying df (and reset index)
        vhh = vhh.copy()
    
    #reset index
    vhh = vhh.reset_index(drop=False)
    
    #free up some memory
    del dist_dict
    
    #save checkpoint
    vhh.to_csv(f'{tmp_dir}/intermediate.csv')
    
    print(f'Finished merges, starting quality filtering.', flush=True)
        
    #############
    ## filter individual clusterings
    #############
    
    id_cols = [col for col in vhh.columns if 'clone_id' in col]
    #skip filtering if all columns will be kept
    if args.score_fraction < 1:
        
        dist_param_map = {col:tuple(cdr3_dist_combos[int(col.split('_')[3])]) for col in id_cols}
        score_df = weighted_ANARCI_silhouette_fast(vhh, id_cols, dist_param_map, args.sample_N_pairs, pool, ncpus)

        score_df = pd.DataFrame(score_df,columns=['name','score']).sort_values(by='score',ascending=False)
        score_df = score_df.head(int(len(score_df)*args.score_fraction))
        keep_cols = score_df['name'].tolist()
                
        print(f'{len(keep_cols)} of {len(id_cols)} clusterings pass the score cutoff ({score_df["score"].min()}).',flush=True)
    else:
        keep_cols = id_cols
        print(f'All clusterings pass the score cutoff.',flush=True)
          
    #############
    ## meta clustering
    #############
    
    #group all VHHs that are always clustered together
    for i,(g,group) in enumerate(vhh.groupby(by=keep_cols,sort=False)):
        vhh.loc[group.index,'fine_clone_id'] = i
    print(f'{len(vhh["fine_clone_id"].unique())} tight clusters found.',flush=True)
    
    #run final coarse clustering with user-specified parameters
    
    #get distances between fine clusters
    #only need to compare one from each group because all within group are the same
    group_reps = vhh.drop_duplicates('fine_clone_id').copy()
    #run clustering
    for method in args.meta_method:
        links = linkage(group_reps[keep_cols].to_numpy(copy=True),method=method,metric='hamming')
        for min_id in tqdm(args.meta_min_id):
            labels = fcluster(links,criterion='distance',t=1-min_id)
            group_reps.loc[:,f'meta_clone_id_{method}_{min_id}'] = labels
            
            print(f'{len(group_reps[f"meta_clone_id_{method}_{min_id}"].unique())} families found using {method} linkage at a {round(min_id*100,1)}% identity cutoff.',flush=True)
    
    meta_id_cols = [col for col in group_reps.columns if 'meta_clone_id' in col]
    
    #add results to vhh dataframe
    print(f'{len(vhh)} VHHs before merge.',flush=True)
    vhh = vhh.merge(group_reps[['fine_clone_id']+meta_id_cols],on='fine_clone_id',how='left')
    print(f'{len(vhh)} VHHs after merge.',flush=True)
    
    vhh = vhh.astype(dtype={col:int for col in id_cols+meta_id_cols})
    
    #clean up
    subprocess.run(['rm', '-r', tmp_dir])
    
    return vhh,id_cols,meta_id_cols

if __name__ == '__main__':
    args = get_args()
    
    #set up for parallelization
    ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    pool = Pool(processes=ncpus)
    print(f'Connected to pool with {ncpus} cpus.',flush=True)
    
    #parse output folder anf file names
    if args.out_file[0] != '/':
        args.out_file = './'+args.out_file
    out_dir = '/'.join(args.out_file.split('/')[:-1])+'/'
    print('out_dir = ',out_dir)
    out_fname = args.out_file.split('/')[-1]
    
    #read all files into big dataframe (this will have duplicates)
    files = glob(f'{args.in_dir}*db-pass.tsv')
    if len(files) == 0:
        print("Can't find tsv files - looking one dir deeper.")
        files = glob(f'{args.in_dir}*/*db-pass.tsv')
        
    data = collect_seqs(files) #get sequence info from user-specificed folder
        
    #define functions for aggregation
    agg_funcs = {}
    for c in data.columns:
        #take the first of each VHH group for descriptor columns
        if c != 'duplicate_count':
            agg_funcs[c] = lambda x: x.iloc[0]
        #sum the duplicate count
        else:
            agg_funcs[c] = np.sum
    
    data,vhh = annotate_and_filter_seqs(args,data,pool,ncpus,agg_funcs) #get translations, CDR anotations, and filter out bad/singleton sequences
    vhh,id_cols,meta_id_cols = metacluster(args,vhh,out_dir,out_fname,pool,ncpus) #get cluster labels
    
    #remove unnecessary column to save space
    del vhh['anarci_parse']
    
    #add clustering results to main dataframe
    print(f'{len(data)} sequences before merge.',flush=True)
    if args.keep_intermediates:
        data = data.merge(vhh[['VHH','CDR1','CDR2','CDR3','scoper_group']+id_cols+meta_id_cols],on='VHH',how='left')
    else:
        data = data.merge(vhh[['VHH','CDR1','CDR2','CDR3']+meta_id_cols],on='VHH',how='left')
    print(f'{len(data)} sequences after merge.',flush=True)
    data.to_csv(args.out_file)
    
    #close parallel pool
    pool.close()