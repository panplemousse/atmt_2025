#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=1:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_sk.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit



# TRANSLATE
python translate.py \
    --cuda \
    --input sk-en/data/raw/test.sk \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output sk-en/output.txt \
    --bleu \
    --reference sk-en/data/raw/test.en \
    --max-len 300
