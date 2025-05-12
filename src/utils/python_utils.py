# a variety of utility functions imported by many of the scripts in this folder

from Bio.Seq import Seq
import re
import numpy as np
import subprocess
from multiprocessing import Pool
from copy import deepcopy
from tqdm import tqdm
import os
import pandas as pd
from Levenshtein import distance
import random
from anarci import anarci
import subprocess
from glob import glob
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import SpectralClustering

# Snip bp from the end if it doesn't "fit"
def translate_VHH(dna_data):
    clean_dna = re.sub("^ACTG", "", dna_data)
    # Check that length is appropriate
    clean_dna = clean_dna[:len(clean_dna) - (len(clean_dna)%3)]
    return str(Seq(clean_dna).translate())[10:] #trim off first 10aa because they're not relevant
  
def CDR3_dist(s1,s2):
    #dist = 1 if s1 and s2 should not be grouped
    #dist = 0 if they should be grouped
    
    if len(s1) != len(s2):
        return 1
    
    #require perfect match for small CDR3s
    if len(s1) <= 6:
        return s1 != s2
    
    #require >80% match for medium length CDR3s
    if (len(s1) > 6) and (len(s1) < 10):
        return np.mean(np.array(list(s1)) == np.array(list(s2))) <= 0.8
        
    #require >60% match for long CDR3s
    if len(s1) >= 10:
        return np.mean(np.array(list(s1)) == np.array(list(s2))) <= 0.6
        
def CDR3_comp(s1,s2):
    #dist = %difference

    if len(s1) != len(s2):
        return np.inf
        
    dist = np.mean(np.array(list(s1)) != np.array(list(s2)))
    
    #require perfect match for small CDR3s
    if len(s1) <= 6:
        if dist == 0:
            return dist
        return np.inf
    
    #require >80% match for medium length CDR3s
    if (len(s1) > 6) and (len(s1) < 10):
        if dist < 0.2:
            return dist
        return np.inf
        
    #require >60% match for long CDR3s
    if len(s1) >= 10:
        if dist < 0.4:
            return dist
        return np.inf
        
def CDR3_nmut(s1,s2):
    #assumes length of s1 and s2 are the same
    
    dist = np.mean(np.array(list(s1)) != np.array(list(s2)))
    
    #allow 80% homology  or better, always allowing at least one AA substitution
    allowed = max(1, np.floor(0.2*len(s1)))
    
    if dist <= allowed:
        return dist
    else:
        return np.inf
    
def CDR3_comp_similarAA(s1,s2):

    allowed = {
        #aromatic
        'Y':{'F','W'},
        'F':{'Y','W'},
        'W':{'Y','F'},
        
        #small (include CYS b/c disulfide unlikely in CDR)
        'A':{'S','T','C','G'},
        'S':{'A','T','C','G'},
        'T':{'S','A','C','G'},
        'C':{'A','T','S','G'},
        'G':{'A','T','S','C'},
        
        #polar/negative
        'D':{'E','Q','N'},
        'E':{'D','Q','N'},
        'Q':{'E','D','N'},
        'N':{'E','Q','D'},
        
        #positive
        'K':{'R'},
        'R':{'K'},
        
        #hydrophobic
        'I':{'M','L','V'},
        'M':{'I','L','V'},
        'L':{'M','I','V'},
        'V':{'M','L','I'},
        
        #weird
        'H':set(),
        'P':set()
        
    }

    if len(s1) != len(s2):
        return np.inf
        
    dist = np.mean(np.array(list(s1)) != np.array(list(s2)))
    
    #require slightly imperfect match for small CDR3s
    if len(s1) <= 6:
        if dist == 0: #if there is a perfect match, dist is 0
            return dist
        else: #if not perfect, allow 2 AA substitutions if it's to a similar AA
            tot = 0
            for i in range(len(s1)):
                
                #ignore perfect matches
                if s1[i]==s2[i]:
                    continue
                #count number of allowed substitutions
                elif s1[i] in allowed[s2[i]]:
                    tot += 1
                #infinite dist if any non-allowed substitutions
                else:
                    return np.inf
            #allow 2 subs if each is allowed
            if tot <= 2:
                return tot
        return np.inf
    
    #require >80% match for medium length CDR3s
    if (len(s1) > 6) and (len(s1) < 10):
        if dist < 0.2:
            return dist
        return np.inf
        
    #require >60% match for long CDR3s
    if len(s1) >= 10:
        if dist < 0.4:
            return dist
        return np.inf
        
def frac_mismatch(s1,s2):
    #assumes s1 and s2 are the same length
    return np.mean(np.array(list(s1)) != np.array(list(s2)))
    
def avg_CDR3_mismatch(fam1,fam2):
    #if comparing a family to itself, return 0
    if len(set(fam1).difference(set(fam2))) == 0:
        return 0
        
    #reduce each family to 1000 random members to speed up calculation
    if len(fam1) > 1000:
        fam1 = random.choices(fam1,k=1000)
    if len(fam2) > 1000:
        fam2 = random.choices(fam2,k=1000)
    
    tot = 0
    n = 0
    for seq1 in fam1:
        for seq2 in fam2:
            n += 1
            #if same length, then it's worth comparing
            if len(seq1) == len(seq2):
                tot += frac_mismatch(seq1,seq2)
            #otherwise say that they are 100% different
            else:
                tot += 1
    return tot/n
        
def dist_func_converter(func_name):
    if func_name == 'CDR3_comp_similarAA':
        return CDR3_comp_similarAA
    elif func_name == 'CDR3_comp':
        return CDR3_comp
    elif func_name == 'CDR3_nmut':
        return CDR3_nmut
    elif func_name == 'frac_mismatch':
        return frac_mismatch

def run_and_clean(cmd):
    code = subprocess.run(cmd).returncode
    subprocess.run(['rm', '-r', cmd[-1]])
    return code
    
def CDR3_dist_row(seq,seqs,func):
    return [func(seq,s1) for s1 in seqs]
    
def aggregate_group(group,count_cols):
    group = group.sort_values(by=['VHH_duplicate_count','duplicate_count'],ascending=False)
    out = group.head(1)
    out[count_cols] = group.loc[:,count_cols].sum(axis=0)
    assert len(out)==1
    return out
    
def run_gfold(q,init_file,final_file,out_file):
    cmd = ['singularity','exec','/cluster/tufts/cowenlab/wwhite06/packages/gfold_1.1.4--gsl1.16_1.sif',
           'gfold','diff',
           '-norm','Count',
           '-s1',init_file,
           '-s2',final_file,
           '-suf','.tsv',
           '-o',out_file,
           '-sc',str(q)]
    gfold_result = subprocess.run(cmd)
    assert gfold_result.returncode == 0
    
def aggregate_and_calculate_fc(data,agg_funcs,agg_col,pair_df,gfold_quantiles,pool):
    
    #aggregate on requested column
    print(f'Aggregating on {agg_col}.',flush=True)
    result = data.groupby(by=agg_col,sort=False).aggregate(agg_funcs)
    result[agg_col] = result.index #move sequence back to a column
    result.index = range(len(result)) #reset index
    print(f'{len(result)} unique {agg_col} values.',flush=True)
    
    #run GFOLD on aggregated data
    os.mkdir('tmp')
    os.mkdir('tmp/inputs')
    os.mkdir('tmp/outputs')
    
    #create datasets to compare
    datasets = set(pair_df['initial'].tolist()+pair_df['final'].tolist())
    short = {}
    for ds_idx,ds in enumerate(datasets):
        count_cols = ds.split('+')
        # short_name = '+'.join([x.split('_')[0] for x in count_cols])
        short[ds] = ds_idx#short_name
        
        for_gfold = result[['sequence_id']+count_cols].copy()
        for_gfold['sum'] = for_gfold[count_cols].sum(axis=1)
        
        #add fake columns for GFOLD
        for_gfold[['2','4','5']] = np.nan
        
        #rearrange columns to match GFOLD:
        #GeneSymbol, GeneName, Read Count, Gene exon length, RPKM
        for_gfold = for_gfold[['sequence_id','2','sum','4','5']]
        for_gfold.to_csv(f'tmp/inputs/{ds_idx}_{agg_col}.tsv',sep='\t',index=False,header=False)
        
    #run GFOLD in parallel for each user-specified quantile
    pool.starmap(run_gfold, ((q,
                              f'tmp/inputs/{short[row["initial"]]}_{agg_col}',
                              f'tmp/inputs/{short[row["final"]]}_{agg_col}',
                              f'tmp/outputs/{row["name"]}_{agg_col}_{str(q).split(".")[1]}.tsv') for q in gfold_quantiles for i,row in pair_df.iterrows()),
                 chunksize=1)
    
    #collect GFOLD results
    for q in gfold_quantiles:
        q_str = str(q).split(".")[1]
        
        for name in pair_df['name']:
            gfold_name = f'gfold{q_str}_{name}'
            gfold = pd.read_csv(f'tmp/outputs/{name}_{agg_col}_{q_str}.tsv',sep='\t',header=9,
                                names=['sequence_id','GeneName',gfold_name,'E-FDR','log2fdc','1stRPKM','2ndRPKM'])
            gfold = gfold[['sequence_id',gfold_name]]

            result = result.merge(gfold,on='sequence_id',how='outer')
        print(f'{len(result)} unique {agg_col} values after GFOLD.',flush=True)
    
    #calculate raw log-fold-change values
    for i,row in pair_df.iterrows():
        init_cols = row['initial'].split('+')
        init_count = result[init_cols].sum(axis=1)
        init_frac = init_count/init_count.sum()
        
        final_cols = row['final'].split('+')
        final_count = result[final_cols].sum(axis=1)
        final_frac = final_count/final_count.sum()
        
        if (max(final_count.sum(),init_count.sum())) == 0:
            print(row['initial'], row['final'], flush=True)
        
        fudge_factor = 1/(max(final_count.sum(),init_count.sum())) #use this to avoid divide-by-zero errors
        
        result = result.assign(**{f'enrich_{row["name"]}': np.log2((final_frac+fudge_factor)/(init_frac+fudge_factor))})
    
    #clean up tmp folder    
    subprocess.run(['rm','-r','tmp'])
    
    #remove unnecessary columns
    for col in ['X','duplicate_count','VHH_duplicate_count','dist_nearest','centroid','centroid_id','family_homology_CDR3','is_outlier']:
        if col in result.columns:
            del result[col]
    
    return result
            
def run_preseq(data,pair_df,out_dir,suffix):
    
    #get all sets of combined samples to run preseq on
    combos = set(pair_df['initial'].tolist()+pair_df['final'].tolist())
    
    for combo in tqdm(combos):
        
        columns = combo.split('+')
        name = '+'.join(set([col.split('_')[-1] for col in columns]))
        
        #add columns together and write count file
        counts = data[columns].sum(axis=1)
        with open(f'{out_dir}/{name}_{suffix}_counts.txt','w') as f:
            for c in counts[counts>0]:
                f.write(f'{c}\n')
            
        #run preseq
        result = subprocess.run(['preseq', 'c_curve',
                                '-V', f'{out_dir}/{name}_{suffix}_counts.txt',
                                '-o', f'{out_dir}/{name}_{suffix}_curve.txt',
                                '-s', '100'])
        
        if result.returncode != 0:
            print(f'{name}_curve preseq failed!')

def run_preseq_assembly(data,sample_df,pair_df,out_dir,kind,pool,
                        seq_col,fam_col,
                        start_seq,end_seq,max_start_scan,max_end_scan,min_start_match,min_end_match,
                        tmp_dir,mmseqs_db,min_seq_id):
    
    if kind not in {'vhh','fam'}:
        raise Exception('Invalid kind in preseq analysis.')
    
    #get all sets of combined samples to run preseq on
    combos = set(pair_df['initial'].tolist()+pair_df['final'].tolist())
    
    for combo in tqdm(combos):
        
        columns = combo.split('+')
        
        #get names of files to load
        fnames = [sample_df.loc[sample_df['name']==col,'file'].tolist()[0] for col in columns]
        #merge unfiltered count data onto main data
        count_info = data[['sequence_id','sequence',seq_col,fam_col]].copy()
        for file in fnames:
            new_df = pd.read_csv(file,sep='\t')[['sequence','duplicate_count']]
            count_info = count_info.merge(new_df,on='sequence',how='outer',suffixes=['',file.split('/')[-1].split('_')[0]])
        #get total counts across all samples
        count_cols = [col for col in count_info.columns if 'duplicate_count' in col]
        count_info['total'] = count_info[count_cols].sum(axis=1)
        
        #get translations of sequences without translation
        missing_locs = count_info[seq_col].map(lambda x: type(x)!=str)
        print(columns, missing_locs.sum(), len(count_info), flush=True)
        dna_seqs = count_info.loc[missing_locs,'sequence']
        aa_df = pd.DataFrame(pool.starmap(translate_VHH_end_scan,
                                          [(seq, start_seq, end_seq, max_start_scan, max_end_scan) for seq in dna_seqs],
                                          chunksize=5000),
                            columns=['sequence',seq_col,'start_match','start_i','start_f','end_match','end_i'])
        #find and label bad seqs
        has_stop = aa_df[seq_col].map(lambda x: '*' in x)
        bad_start = aa_df['start_match'] < min_start_match
        bad_end = aa_df['end_match'] < min_end_match
        bad_seq = has_stop|bad_start|bad_end
        aa_df.loc[bad_seq,seq_col] = ''
        
        #fill in missing translations
        count_info.loc[missing_locs,seq_col] = aa_df[seq_col].values
        #remove failed translations
        count_info = count_info.loc[count_info[seq_col]!='',:]
        
        #aggregate by vhh
        agg_funcs = {
            'sequence_id':lambda x: x.iloc[0],
            fam_col:lambda x: x.iloc[0],
            'total':np.sum,
        }
        count_info = count_info[['sequence_id',fam_col,seq_col,'total']].groupby(by=seq_col,sort=False).aggregate(agg_funcs)
        #bring back VHH as a column instead of index
        count_info = count_info.reset_index()
        count_info.rename({'index':seq_col})
        
        #find families if necessary
        if kind == 'fam':
            
            #need to aldo fill in missing fam info
            missing_fams = count_info[fam_col].map(lambda x: type(x)!=str)
            print(missing_fams.sum(),flush=True)
            
            #add seq_ids to missing rows
            count_info.loc[missing_fams,'sequence_id'] = [f'N-{i}' for i in range(missing_fams.sum())]
            
            #write mmseqs input fasta
            with open(f'{tmp_dir}/tmp_mmseqs_query.fasta','w') as f:
                for i,row in count_info.loc[missing_fams,:].iterrows():
                    f.write(f'>{row["sequence_id"]}\n{row[seq_col]}\n')
                    
            #run mmseqs to find closest family
            result = subprocess.run(['/cluster/tufts/cowenlab/emosel01/condaenv/vhh/bin/mmseqs',
                                     'easy-search', f'{tmp_dir}/tmp_mmseqs_query.fasta', mmseqs_db, f'{tmp_dir}/mmseqs_results', tmp_dir,
                                     '-s', '6', #sensitivity
                                     '--local-tmp', tmp_dir,
                                     '--format-output', 'query,target,pident',
                                     '--max-seq-id','1.0', #do not do any redundancy filtering
                                     '--max-seqs', '100', #look at top 100 seqs per query
                                     '-v', '0'])
            assert result.returncode == 0
                            
            mmseqs_df = pd.read_csv(f'{tmp_dir}/mmseqs_results',names=['query','target','pident'],sep='\t')
            fam_idx = 0
            for seq_id,group in mmseqs_df.groupby('query'):
                best_idx = group['pident'].idxmax()
                if group.loc[best_idx,'pident'] >= min_seq_id: #close enough match
                    match_id = count_info.loc[count_info['sequence_id']==group.loc[best_idx,'target'],fam_col] #get family of best match
                    assert len(match_id) == 1
                    count_info.loc[count_info['sequence_id']==seq_id,fam_col] = match_id #set clone_id of the query sequence 
                else: #not a good match to any family
                    count_info.loc[count_info['sequence_id']==seq_id,fam_col] = f'N-{fam_idx}' #give new clone_id
                    fam_idx += 1 #increment new clone_id counter
                    
            #aggregate by family
            count_info = count_info[['clone_id','total']].groupby(by='clone_id',sort=False).aggregate(np.sum)
            
            #clean up
            for item in glob(f'{tmp_dir}/*'):
                if item != mmseqs_db:
                    result = subprocess.run(['rm', '-r', item])
                    assert result.returncode == 0
        
        name = '+'.join(set([col.split('_')[-1] for col in columns]))
        
        #add columns together and write count file
        with open(f'{out_dir}/{name}_{kind}_counts.txt','w') as f:
            for c in count_info['total']:
                f.write(f'{c}\n')
            
        #run preseq
        result = subprocess.run(['preseq', 'c_curve',
                                '-V', f'{out_dir}/{name}_{kind}_counts.txt',
                                '-o', f'{out_dir}/{name}_{kind}_curve.txt',
                                '-s', '100'])
        
        if result.returncode != 0:
            print(f'{name}_curve preseq failed!')
            
def translate_VHH_end_scan(dna,start_aa,end_aa,start_cap,end_cap):
    
    start_aa = np.array(list(start_aa))
    start_len = len(start_aa)
    
    #check all frames
    best_match_start = 0
    best_aa = ''
    best_i_start = np.nan
    best_f = np.nan
    for f in range(3):
        
        L = (len(dna)-f) - (len(dna)-f)%3
        aa = np.array(Seq(dna[f:L+f]).translate())
        
        #slide start_aa along translated seq to find best match (stop after first 40aa)
        for i in range(1,min(start_cap,len(aa))):
            
            #take the last i residues of start_aa (or the whole thing if i>start_len)
            start_segment = start_aa[-min(i,start_len):]
            
            #take the first i residues of aa (or a start_len stretch inside aa)
            aa_segment = aa[max(0,i-start_len):i]
            
            assert len(start_segment) == len(aa_segment)
            
            #count mismatches + non-overlap
            match = np.sum(start_segment == aa_segment)
            
            if match > best_match_start:
                best_match_start = match
                #add on the first piece of start_aa if best overlap is not a full overlap
                best_aa = np.concatenate([start_aa[:max(start_len-i,0)],aa[max(i-start_len,0):]],axis=0)
                best_i_start = i
                best_f = f
    
    aa = best_aa
    aa_len = len(aa)
        
    end_aa = np.array(list(end_aa))
    end_len = len(end_aa)
    
    best_match_end = 0
    best_aa = ''
    best_i_end = np.nan
    #given forward translation, find end point by sliding end_aa along the translation backwards
    for i in range(1,min(end_cap,len(aa))):

        #take the first i residues of end_aa (or the whole thing if i>end_len)
        end_segment = end_aa[:min(i,end_len)]

        #take the last i residues of aa (or a end_len stretch inside aa)
        aa_segment = aa[-i:min(aa_len-i+end_len,aa_len)]

        assert len(end_segment) == len(aa_segment)

        #count mismatches + non-overlap
        match = np.sum(end_segment == aa_segment)

        if match > best_match_end:
            best_match_end = match
            #add on the last piece of end_aa if best overlap is not a full overlap
            best_aa = np.concatenate([aa[:min(aa_len-i+end_len,aa_len)],end_aa[min(i,end_len):]],axis=0)
            best_i_end = i
                    
    return dna, ''.join(best_aa), best_match_start, best_i_start, best_f, best_match_end, best_i_end
            
def get_anarci_alignment(seq):
    
    sequences = [('id',seq)]
    numbering, _, _ = anarci(sequences, scheme="imgt", output=False, allowed_species=['alpaca'])
    
    assert len(numbering)==1 #only one sequence result
    
    #fail if no parse can be determined
    if numbering[0] is None:
        return seq, (np.array([]),np.array([])), '*', '*', '*', False
    #fail if parse contains multiple domains
    if len(numbering[0]) != 1:
        return seq, (np.array([]),np.array([])), '*', '*', '*', False
        
    start_idx = numbering[0][0][1]
    end_idx = numbering[0][0][2]
    numbering = numbering[0][0][0]
        
    #fill in the numbering to get to 128aa (this is canonical VHH length for IMGT)
    numbering = numbering + [((i,' '),'-') for i in range(numbering[-1][0][0]+1,129)]
    
    #add back in any residues trimmed off at the start
    numbering = [((i-start_idx+1,' '),seq[i]) for i in range(start_idx)] + numbering
    
    #add back in any residues trimmed off at the end
    numbering = numbering + [((128+i-end_idx,' '),seq[i]) for i in range(end_idx+1,len(seq))]
    
    seq_array = np.array([pos[1] for pos in numbering])
    num_array = np.array([pos[0][0] for pos in numbering])
    
    return (seq, (seq_array, num_array),
            ''.join(seq_array[(num_array>=27)&(num_array<=38)]).replace('-',''), #CDR1
            ''.join(seq_array[(num_array>=56)&(num_array<=65)]).replace('-',''), #CDR2
            ''.join(seq_array[(num_array>=105)&(num_array<=117)]).replace('-',''),#CDR3
            True)
            
def weighted_anarci_dist(CDR_list1,CDR_list2,weight_scheme,max_len_diffs):
    
    #set dist to 100% if either of the pair had no ANARCI parse
    if (CDR_list1[0] == '*') or (CDR_list2[0] == '*'):
        return 1
        
    #set dist to 100% if any CDR pair is too different in length
    for i in range(len(CDR_list1)-1,-1,-1): #do this in reverse order because most likely to have CDR3 diff so will return faster
        if abs(len(CDR_list1[i]) - len(CDR_list2[i])) > max_len_diffs[i]:
            return 1
    
    dist_tot = 0
    weight_tot = 0
    for i in range(len(CDR_list1)):
        dist = distance(CDR_list1[i],CDR_list2[i]) #edit distance between CDR pair
        dist_tot += dist*weight_scheme[i] #add weighted dist to total
        weight_tot += np.max([len(CDR_list1[i]),len(CDR_list2[i])])*weight_scheme[i] #add weight to total
        
    return dist_tot/weight_tot #compute weighted percent identity
    
def weighted_anarci_dist_with_penalty(CDR_list1,CDR_list2,weight_scheme,max_len_diffs,penalty):
    
    #set dist to 100% if either of the pair had no ANARCI parse
    if (CDR_list1[0] == '*') or (CDR_list2[0] == '*'):
        return 1
        
    #set dist to 100% if any CDR pair is too different in length
    for i in range(len(CDR_list1)-1,-1,-1): #do this in reverse order because most likely to have CDR3 diff so will return faster
        if abs(len(CDR_list1[i]) - len(CDR_list2[i])) > max_len_diffs[i]:
            return 1
    
    dist_tot = 0
    weight_tot = 0
    for i in range(len(CDR_list1)):
        dist = distance(CDR_list1[i],CDR_list2[i]) #edit distance between CDR pair
        dist_tot += dist*weight_scheme[i] #add weighted dist to total
        weight_tot += np.max([len(CDR_list1[i]),len(CDR_list2[i])])*weight_scheme[i] #add weight to total
    
    tot_penalty = 0
    for i in range(len(CDR_list1)):
        tot_penalty += penalty*abs(len(CDR_list1[i]) - len(CDR_list2[i]))
        
    return min(1, dist_tot/weight_tot + tot_penalty) #compute weighted percent identity
    
def ANARCI_dist_row(CDR_list1,CDR_list_list,weight_scheme,max_len_diffs):
    return [weighted_anarci_dist(CDR_list1,CDR_list2,weight_scheme,max_len_diffs) for CDR_list2 in CDR_list_list]
    
def ANARCI_dist_with_penalty_row(CDR_list1,CDR_list_list,weight_scheme,max_len_diffs,penalty):
    return [weighted_anarci_dist_with_penalty(CDR_list1,CDR_list2,weight_scheme,max_len_diffs,penalty) for CDR_list2 in CDR_list_list]
    
def weighted_anarci_dist_index(idx1,idx2,df,weight_scheme,max_len_diffs):
    cdr_list1 = df.loc[idx1[0],['CDR1','CDR2','CDR3']].tolist()
    cdr_list2 = df.loc[idx2[0],['CDR1','CDR2','CDR3']].tolist()
    return weighted_anarci_dist(cdr_list1,cdr_list2,weight_scheme,max_len_diffs)

def recursive_mmseqs_split(group, max_group_size, min_id, id_step, tmp_dir, ncpus):
    
    group_index = group.index.copy()
    
    #base case (if smaller than max allowed)
    if len(group) <= max_group_size:
        group['mmseqs_id'] = group['sequence_id'].values[0] #arbitrarily use first one
        return group, min_id-id_step
    #otherwise split into groups recursively with mmseqs
    else:
        print(len(group),min_id,flush=True)
        #write fasta file
        with open(f'{tmp_dir}/mmseqs_in.fasta','w') as f:
            for i,row in group.iterrows():
                f.write(f">{row['sequence_id']}\n{row['VHH']}\n") #Index here is the intermediate_id
                
        #run mmseqs clustering
        result = subprocess.run(['/cluster/tufts/cowenlab/emosel01/condaenv/vhh/bin/mmseqs', 'easy-cluster',
                                f'{tmp_dir}/mmseqs_in.fasta', f'{tmp_dir}/mmseqs_out', f'{tmp_dir}/tmp',
                                 '-s', '7.5', #sensitivity set to max
                                 '-c', str(max(0,(min_id - 0.2))), #set this lower than seq-id cutoff so it doesn't have an impact
                                 '--min-seq-id', str(min_id), #only cluster seqs more similar than user input
                                 '-e', 'inf', #max e-value to be clustered (set to keep all)
                                 '-v', '1', #verbosity
                                 '--cluster-steps', '3', #use faster cascaded clustering method
                                 '--seq-id-mode', '0', #use alignment length to normalize
                                 '--cluster-mode', '0', #use greedy setcover algorithm
                                 '--alignment-mode', '3', #get percent identity, rather than alignment score
                                 '--max-seqs', '1000', #get top 1000 matches to each query
                                 '--gap-open', '20', #set high gap-opening penalty to prevent excessive gaps
                                 '--gap-extend', '3',#set high gap-extend penalty to prevent excessive gaps
                                 '--similarity-type', '2', #use sequence identity, not score
                                 '--threads', str(ncpus)]) #how many cpus to use
        assert result.returncode == 0
        
        #read in clusters
        mmseqs_clusters = pd.read_csv(f'{tmp_dir}/mmseqs_out_cluster.tsv',names=['mmseqs_id','sequence_id'],sep='\t')
        
        #remove old clusters and add new ones
        group = group.drop(columns=['mmseqs_id'],errors='ignore')
        group = group.merge(mmseqs_clusters,on='sequence_id',how='left')
        group.index = group_index #this preserves the original index
        
        #clean up
        for f in glob(f'{tmp_dir}/*'):
            subprocess.run(['rm', '-r', f])

        subgroups = []
        max_min_id = 0
        for _,subgroup in group.groupby('mmseqs_id'):
            split, sg_min_id = recursive_mmseqs_split(subgroup, max_group_size, min_id+id_step, id_step, tmp_dir, ncpus)
            subgroups.append(split)
            max_min_id = max(max_min_id,sg_min_id)
            
        return pd.concat(subgroups,axis=0), max_min_id

def group_meta_dist(clust1,clust2,method):
    dists = cdist(clust1, clust1, metric='hamming').flatten()
    if method == 'single':
        return np.min(dists)
    if method == 'average':
        return np.mean(dists)
    if method == 'complete':
        return np.max(dists)
    raise Exception(f'{method} is not a valid method.')

def recursive_spectral_clustering(group, id_cols, min_id, agg_method, spec_method, spec_ncomp, min_sim):
    
    dists = pdist(group.loc[:,id_cols].values,metric='hamming')
    if agg_method == 'single':
        dist_summary = np.min(dists)
    elif agg_method == 'average':
        dist_summary = np.mean(dists)
    elif agg_method == 'complete':
        dist_summary = np.max(dists)
    
    #base case (if distances are small enough, or only one member of group)
    if (dist_summary <= (1-min_id)) or (len(group)==1):
        group['meta_clone_id'] = group['spec_clone_id'].to_numpy()[0] #arbitrarily use first one
        # print('Done:', len(group),dist_summary,flush=True)
        return group
    #otherwise split into groups recursively with spectral clustering
    else:
        # print('Splitting:', len(group),dist_summary,flush=True)
        
        #remove old clusters and add new ones
        group = group.drop(columns=['meta_clone_id'],errors='ignore')
        
        clusterer = SpectralClustering(n_clusters = 2,
                                       n_components = min(spec_ncomp,len(group)-1),
                                       affinity = 'precomputed',
                                       assign_labels = spec_method,
                                       n_jobs=-1)
        group['meta_clone_id'] = clusterer.fit_predict(squareform(1-dists+min_sim))

        subgroups = []
        for _,subgroup in group.groupby('meta_clone_id'):
            split = recursive_spectral_clustering(subgroup, id_cols, min_id, agg_method, spec_method, spec_ncomp, min_sim)
            subgroups.append(split)

        return pd.concat(subgroups,axis=0)
        
def add_header(df, header_names, sample_df):
    header = []
    for col in df.columns:
        #for enrighment and gfold columns, look for label inside raw data column names
        if col.startswith('enrich') or col.startswith('gfold'):
            name = col.split('_')[1]
            matches = sample_df[sample_df['no_rep']==name]
        #for others check for exact match to find raw data columns
        else:
            matches = sample_df[sample_df['name']==col]
            
        if len(matches): #this means this is a column we want to label because it's has quantitative data
            
            assert len(matches['no_rep'].unique()) == 1 #make sure we're getting a single sample's worth of data
            
            match_names = matches[header_names].iloc[0,:]
            match_names[match_names.isna()] = ''
            header.append(tuple(match_names.tolist()+[col]))
            
        else: #fill in blank info for columns with sequence descriptors
            header.append(tuple(['']*len(header_names)+[col]))
            
    df.columns = pd.MultiIndex.from_tuples(header,names=header_names+['full_name'])
    return df

def norm_dist(seq1,seq2):
    return distance(seq1,seq2)/max(len(seq1),len(seq2))
                        
def cluster_ratio_score(seqs,max_diff,pool):
    if len(seqs) == 1:
        return 0
    
    dists = np.array(pool.starmap(norm_dist,((s1,s2) for i,s1 in enumerate(seqs[:-1]) for s2 in seqs[i+1:]),chunksize=1000))
    tot_good = np.sum(dists<=max_diff)
    tot_bad = np.sum(dists>max_diff)
        
    return tot_good/(tot_bad+1)