# DNABERT2 Fine-Tuning for DHS Specificity Prediction

This project focuses on fine-tuning the DNABERT2 model to predict the tissue specificity of DNase I hypersensitive sites (DHSs) across 14 biosamples. The project uses a dataset comprising 733 human biosamples, with DHS metadata and a pre-trained transformer model for the prediction tasks.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Fine-Tuning](#model-fine-tuning)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
The aim of this project is to predict the tissue specificity of DHSs by fine-tuning a DNABERT2 model. DHSs are important regulatory regions in the genome, and their specificity to certain tissues can provide insights into gene regulation and potential disease mechanisms.

## Dataset
The dataset used in this project includes:
- DHS metadata for 733 biosamples.
- NMF (Non-negative Matrix Factorization) basis and coefficient matrices.
- Binary matrix indicating the presence (1) or absence (0) of DHS peaks in different tissues and cells.

### Data Sources
- **Genome FASTA file**: `hg38.fa`
- **DHS metadata**: `DHS_Index_and_Vocabulary_metadata.tsv`
- **NMF basis array**: `2018-06-08NC16_NNDSVD_Basis.npy`
- **Binary matrix**: `dat_bin_FDR01_hg38.txt`

## Preprocessing
The preprocessing steps include:
1. **Loading and Querying the Reference Genome**: 
   - The `ReferenceGenome` class is used to load the genome data from a FASTA file.
   - DHS sequences are extracted based on chromosomal coordinates and summit positions.
2. **Loading NMF Basis and Coefficients**:
   - The NMF basis array is loaded from a `.npy` file and converted to a DataFrame.
   - DHS metadata is combined with the NMF loadings to form a comprehensive dataset.
3. **Adding Sequence Column**:
   - Sequences for each DHS are added to the DataFrame by querying the reference genome.

## Model Fine-Tuning
The DNABERT2 model is fine-tuned on the dataset to predict the tissue specificity of DHSs. Follow these steps to fine-tune the model:

### 1. Format Your Dataset
Generate three CSV files from your dataset: `train.csv`, `dev.csv`, and `test.csv`. Each file should have the same format with the first row as the header (`sequence, label`) and each following row containing a DNA sequence and a numerical label separated by a comma (e.g., `ACGTCAGTCAGCGTACGT, 1`).

### 2. Fine-Tune DNABERT2
Navigate to the `finetune` directory and set up the necessary environment variables. Then run the fine-tuning script.

#### Single-GPU Training
```bash
cd finetune

export DATA_PATH=./path/to/data/folder  # e.g., ./sample_data
export MAX_LENGTH=100  # Set to 0.25 * your sequence length
export LR=3e-5

python train.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --data_path ${DATA_PATH} \
    --kmer -1 \
    --run_name DNABERT2_${DATA_PATH} \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 5 \
    --fp16 \
    --save_steps 200 \
    --output_dir output/dnabert2 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 100 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False
