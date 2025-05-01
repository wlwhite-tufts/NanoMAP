# VHH Clustering Pipeline

This repository contains code for processing and analyzing VHH (nanobody) antibody sequences.

## Directory Structure

- `pipeline/`: Shell scripts for sequence assembly and preprocessing
- `src/`: Source code for analysis
  - `analysis/`: Scripts for analyzing clustered sequences
  - `clustering/`: Sequence clustering algorithms
    - `anarci/`: ANARCI-based clustering methods
    - `legacy/`: Original hierarchical clustering methods
  - `preprocessing/`: Scripts to prepare raw data
  - `utils/`: Shared utility functions

## Pipeline Overview

1. Assembly and filtering: `pipeline/vhh_assembly_v1.sh`
2. Clustering: Either legacy (`HierarchicalClustering_v3.py` + `CleanCluster_v2.py`) or ANARCI-based methods
3. Analysis: Calculate enrichment using `FinalTabulation_v2.py`