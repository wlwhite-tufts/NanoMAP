import os
import argparse
from glob import glob
import subprocess
import numpy as np
import yaml

#get main dir for repo
repo_path = os.path.dirname(os.path.abspath(__file__))
repo_path = '/'.join(repo_path.split('/')[:-1])

#load relevant variables from YAML
with open(f'{repo_path}/data/user_data.yml', 'r') as f:
    user_data = yaml.safe_load(f)

    presto_path = user_data['paths']['presto_path']
    changeo_path = user_data['paths']['changeo_path']
    igblast_path = user_data['paths']['igblast_path']
    
def get_args():
    
    parser = argparse.ArgumentParser(prog='Assemble and filter reads',
                                     description='Process a directory of fastq files. Assemble forward and reverse reads, apply QC filters. Will pick up at the last saved intermediate step.')
    
    parser.add_argument('--in_dir',type=str,help='The directory that contains the fastq files.')
    parser.add_argument('--out_dir',type=str,help='The directory to put the annotated and filtered files.')
    parser.add_argument('--primer_R1',type=str,help='Path to the fasta file containing the R1 primer definitions.')
    parser.add_argument('--primer_R2',type=str,help='Path to the fasta file containing the R2 primer definitions.')
    parser.add_argument('--R1_suffix',type=str,default='_L001_R1_001.fastq',help='File name ending that indicates a "read 1" sequencing read.')
    parser.add_argument('--R2_suffix',type=str,default='_L001_R2_001.fastq',help='File name ending that indicates a "read 2" sequencing read.')
    parser.add_argument('--save_intermediates',action='store_true',help='Set this flag to save all intermediate assembly and filtering files.')
    
    return parser.parse_args()
    
if __name__ == '__main__':
    
    args = get_args()
    
    #convert everything to absolute paths
    args.in_dir = os.path.realpath(args.in_dir)
    args.out_dir = os.path.realpath(args.out_dir)
    args.primer_R1 = os.path.realpath(args.primer_R1)
    args.primer_R2 = os.path.realpath(args.primer_R2)
    
    #make output folder if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    #find the files to assemble and filter
    R1_files = glob(f'{args.in_dir}/*{args.R1_suffix}')
    R2_files = glob(f'{args.in_dir}/*{args.R2_suffix}')
    assert len(R1_files) == len(R2_files)
    
    #define output order
    order = [
        '_assemble-pass.fastq',
        '_quality-pass.fastq',
        '_MPV_primers-pass.fastq',
        '_MPC_primers-pass.fastq',
        '_collapse-unique.fastq',
        '_atleast-1.fasta',
        '_atleast-1.fmt7',
        '_db-pass.tsv'
        ]
    
    igblast_prefix = f'{igblast_path}/internal_data/alpaca/alpaca_'
    imgt_prefix = f'{igblast_path}/database/alpaca/IG_DNA/imgt_gapped_alpaca_IGH'
    static_cmds = [
        [f'{presto_path}/AssemblePairs.py', 'align', '--coord', 'illumina', '--rc', 'tail', '--maxerror', '0.2', '--outdir', args.out_dir],
        [f'{presto_path}/FilterSeq.py', 'quality', '-q', '20', '--outdir', args.out_dir],
        [f'{presto_path}/MaskPrimers.py', 'score', '-p', args.primer_R1, '--start', '0', '--mode', 'cut', '--pf', 'VPRIMER', '--outdir', args.out_dir],
        [f'{presto_path}/MaskPrimers.py', 'score', '-p', args.primer_R2, '--start', '0', '--mode', 'cut', '--revpr', '--pf', 'CPRIMER', '--outdir', args.out_dir],
        [f'{presto_path}/CollapseSeq.py', '-n', '20', '--inner', '--uf', 'CPRIMER', '--cf', 'VPRIMER', '--act', 'set', '--outdir', args.out_dir],
        [f'{presto_path}/SplitSeq.py', 'group', '-f', 'DUPCOUNT', '--num', '1', '--fasta', '--outdir', args.out_dir],
        [f'{igblast_path}/bin/igblastn', '-germline_db_V', f'{igblast_prefix}V', '-germline_db_D', f'{igblast_prefix}D', '-germline_db_J', f'{igblast_prefix}J',
                '-domain_system', 'imgt', '-ig_seqtype', 'Ig', '-organism', 'alpaca', '-auxiliary_data', f'{igblast_path}/optional_file/alpaca_gl.aux',
                '-outfmt', '7 std qseq sseq btop'],
        [f'{changeo_path}/MakeDb.py', 'igblast', '-r', f'{imgt_prefix}V.fasta', f'{imgt_prefix}D.fasta', f'{imgt_prefix}J.fasta', '--outdir', args.out_dir,
                '--failed', '--format', 'airr', '--extended']
        ]
        
    dynamic_cmds = [
        lambda name: ['-1', f'{args.in_dir}/{name}{args.R1_suffix}', '-2', f'{args.in_dir}/{name}{args.R2_suffix}', '--outname', name],
        lambda name: ['-s', f'{args.out_dir}/{name}_assemble-pass.fastq', '--outname', name],
        lambda name: ['-s', f'{args.out_dir}/{name}_quality-pass.fastq', '--outname', f'{name}_MPV'],
        lambda name: ['-s', f'{args.out_dir}/{name}_MPV_primers-pass.fastq', '--outname', f'{name}_MPC',],
        lambda name: ['-s', f'{args.out_dir}/{name}_MPC_primers-pass.fastq', '--outname', name],
        lambda name: ['-s', f'{args.out_dir}/{name}_collapse-unique.fastq', '--outname', name],
        lambda name: ['-query', f'{args.out_dir}/{name}_atleast-1.fasta', '-out', f'{args.out_dir}/{name}_atleast-1.fmt7'],
        lambda name: ['-i', f'{args.out_dir}/{name}_atleast-1.fmt7', '-s', f'{args.out_dir}/{name}_atleast-1.fasta', '--outname', name, '--log', f'{args.out_dir}/MakeDb_{name}.log']
        ]
    
    #get sample names
    base_names = [name.split('/')[-1].replace(args.R1_suffix,'') for name in R1_files]
    print(base_names, flush=True)
    for name in base_names:
        #make sure reverse version of file exists
        assert os.path.isfile(f'{args.in_dir}/{name}{args.R2_suffix}')
        
        #make a checkpoint file, if one does not exist
        if not os.path.isfile(f'{args.out_dir}/{name}_chkpt.txt'):
            with open(f'{args.out_dir}/{name}_chkpt.txt','w') as f:
                f.write('0')
                
        #check checkpoint
        chk = int(open(f'{args.out_dir}/{name}_chkpt.txt','r').readlines()[0].strip())
        
        #loop through the steps
        for s in range(chk,8):
            
            #remove partially completed files from this step if they exist
            if os.path.isfile(f'{args.out_dir}/{name}{order[s]}'):
                os.remove(f'{args.out_dir}/{name}{order[s]}')
            
            #IgBLAST needs to be run from inside its own directory
            if s == 6:
                os.chdir(igblast_path)
                
            #get and run command
            cmd = static_cmds[s]+dynamic_cmds[s](name)
            result = subprocess.run(cmd)
            assert result.returncode == 0
            
            #update checkpoint
            with open(f'{args.out_dir}/{name}_chkpt.txt','w') as f:
                f.write(str(s+1))
            
            #remove output of previous step, if requested
            #   skip on first one because no previous step exists
            #   skip on IgBlast (s==6) because MakeDb (s==7) needs IgBLAST and SplitSeq (s==5) outputs
            if (not args.save_intermediates) and (s>0) and (s!=6):
                os.remove(f'{args.out_dir}/{name}{order[s-1]}')
                
        #remove leftover files, if requested
        if not args.save_intermediates:
            to_remove = ['{args.out_dir}/{name}_atleast-1.fasta',
                        f'{args.out_dir}/{name}_under-1.fasta',
                        f'{args.out_dir}/{name}_db-fail.tsv',
                        f'{args.out_dir}/MakeDb_{name}.log']
            for file in to_remove:
                if os.path.isfile(file):
                    os.remove(file)
            