# convenience script for setting up batches of raw sequencing files to be assembled and filtered

import argparse
from glob import glob
import os
import subprocess
import numpy as np

parser = argparse.ArgumentParser(prog='FASTQ batches',
                                 description='Split fastq dir into subdirs and create commands to process them.')

parser.add_argument('--in_dir',type=str,help='The directory that contains the fastq(.gz) files.')
parser.add_argument('--cmd_file',type=str,help='Name for the command file to write.')
parser.add_argument('--N',type=int,help='The number of samples to put in each batch. Each sample has 2 fastq files.')
parser.add_argument('--primer_F',type=str,default='/cluster/tufts/cowenlab/wwhite06/vhh/primers/AlpVHH_R1_primers_05Feb24.fasta',
                    help='The forward primer definitions.')
parser.add_argument('--primer_R',type=str,default='/cluster/tufts/cowenlab/wwhite06/vhh/primers/AlpVHH_R2_primers_05Feb24.fasta',
                    help='The reverse primer definitions.')
parser.add_argument('--assembly_script',type=str,default='/cluster/tufts/cowenlab/wwhite06/vhh/scripts/vhh_assembly_v1.sh',
                    help='The script to use when writing assembly commands.')
parser.add_argument('--assembly_dir',type=str,default='assembly',
                    help='The folder name to use as the destination in all commands (do not include full path).')


args = parser.parse_args()

files = glob(args.in_dir+'/*.fastq*') #gets fastq and fastq.gz

real_N = args.N*2 #need to keep forward and reverse reads together

base_dir = args.in_dir + '/..' #go out one dir form fastq

with open(args.cmd_file,'w') as cmd:
    for b in range(int(np.ceil(len(files)/real_N))):
        os.mkdir(args.in_dir+f'/batch{b}')
        for f in files[b*real_N:min(len(files),(b+1)*real_N)]:
            #if not unzipped yet, unzip the file
            if f.endswith('.gz'):
                result = subprocess.run(['gzip', '-d', f])
                assert result.returncode == 0
            #move the file
            result = subprocess.run(['mv', f.replace('.gz',''), args.in_dir+f'/batch{b}/'])
            assert result.returncode == 0
    
        cmd.write(f'/cluster/tufts/cowenlab/wwhite06/vhh/scripts/vhh_assembly_v1.sh '+\
                  f'{args.in_dir}/batch{b} '+\
                  f'{base_dir}/{args.assembly_dir} '+\
                  args.primer_F + ' ' + args.primer_R + '\n')
               
               
               
               
               
               
               
               
               
               
               
               