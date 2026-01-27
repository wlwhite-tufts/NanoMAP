#test batching and assembly/filtering
#python ../processing/make_fastq_batches.py --in_dir fastq/ --cmd_file assembly_cmds.sh --N 2
#cat assembly_cmds.sh | bash

#test clustering
python ../processing/metacluster.py --in_dir tsv --out_file metaclust.csv --meta_min_id 0.25  --meta_method single average complete --max_subgroup_size 200 --sample_N_pairs 10 --recursive_min_id_step 0.15

#test enrichment
python ../analysis/enrichment.py --in_file metaclust.csv --sample_file sample_info.csv --pair_file enrich_pairs.csv

#test scoring
python ../analysis/silhouette_score.py --in_file metaclust.csv --out_file sil_score.csv --label_cols meta_clone_id_single_0.25 meta_clone_id_average_0.25 meta_clone_id_complete_0.25
python ../analysis/phenotype_quality.py --in_file metaclust_gfold_vhh.csv --out_file phen_score.csv --label_cols meta_clone_id_single_0.25 meta_clone_id_average_0.25 meta_clone_id_complete_0.25 --phenotype_cols enrich_A enrich_B enrich_C enrich_D
