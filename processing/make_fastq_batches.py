import argparse
from glob import glob
import os
import subprocess
import numpy as np
import sys

#get main dir for repo
repo_path = os.path.dirname(os.path.abspath(__file__))
repo_path = '/'.join(repo_path.split('/')[:-1])
sys.path.append(repo_path)
from data.user_data import R1_primer_file, R2_primer_file

def get_args(path):
    
    parser = argparse.ArgumentParser(prog='FASTQ batches',
                                     description='Split fastq dir into subdirs within the specified input dir and create commands to assemble and filter them. Any .gz files will be unzipped.')
    
    parser.add_argument('--in_dir',type=str,help='The directory that contains the fastq(.gz) files.')
    parser.add_argument('--cmd_file',type=str,help='Name for the command file to write.')
    parser.add_argument('--N',type=int,help='The number of samples to put in each batch. Each sample has 2 fastq files.')
    parser.add_argument('--primer_R1',type=str,default=f'{R1_primer_file}',help='fasta file with the R1 primer definitions.')
    parser.add_argument('--primer_R2',type=str,default=f'{R2_primer_file}',help='fasta file with the R2 primer definitions.')
    parser.add_argument('--assembly_script',type=str,default=f'{path}/processing/assemble_and_filter_reads.py',
                        help='The script to use when writing assembly commands.')
    parser.add_argument('--assembly_dir',type=str,default='assembly',
                        help='The folder name to use as the destination in all commands.')
                        
    return parser.parse_args()

if __name__ == '__main__':
    
    #get main dir for repo
    repo_path = os.path.dirname(os.path.abspath(__file__))
    repo_path = '/'.join(repo_path.split('/')[:-1])
    
    args = get_args(repo_path)
    
    #find all fastq files in the input dir
    files = glob(args.in_dir+'/*.fastq*')
    real_N = args.N*2 #need to keep forward and reverse reads together
    
    #make assembly dir if it diesn't exist already
    if not os.path.exists(args.assembly_dir):
        os.mkdir(args.assembly_dir)
    
    with open(args.cmd_file,'w') as cmd:
        
        #create each batch
        for b in range(int(np.ceil(len(files)/real_N))):
            
            os.mkdir(args.in_dir+f'/batch{b}') #make the batch dir
            
            #go through the files in this batch
            for f in files[b*real_N:min(len(files),(b+1)*real_N)]:
                #if not unzipped yet, unzip the file
                if f.endswith('.gz'):
                    result = subprocess.run(['gzip', '-d', f])
                    assert result.returncode == 0
                #move the file
                result = subprocess.run(['mv', f.replace('.gz',''), args.in_dir+f'/batch{b}/'])
                assert result.returncode == 0
            
            #write the assembly/filtering command for the batch
            cmd.write(f'python {args.assembly_script} '+\
                      f'--in_dir {args.in_dir}/batch{b} '+\
                      f'--out_dir {args.assembly_dir} '+\
                      f'--primer_R1 {args.primer_R1} '+\
                      f'--primer_R2 {args.primer_R2}\n')