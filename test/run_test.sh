#test batching and assembly/filtering
python ../processing/make_fastq_batches.py --in_dir fastq/ --cmd_file assembly_cmds.sh --N 2
cat assembly_cmds.sh | bash

#test clustering
python ../processing/NanoMAP_meta.py --in_dir tsv --out_file meta_clust.csv --sample_N_pairs 10
python ../processing/NanoMAP_ind.py --in_dir tsv --out_file ind_clust.csv

#test enrichment
python ../analysis/enrichment.py --in_file ind_clust.csv --sample_file sample_info.csv --pair_file enrich_pairs.csv

#test scoring
python ../analysis/silhouette_score.py --in_file ind_clust.csv --out_file sil_score.csv --label_cols clone_id
python ../analysis/phenotype_quality.py --in_file ind_clust_gfold_vhh.csv --out_file phen_score.csv --label_cols clone_id --phenotype_cols enrich_A enrich_B enrich_C enrich_D
