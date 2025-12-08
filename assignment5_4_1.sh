#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=length_norm_assignment5.out

module load a100
module load miniforge3
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# TRANSLATE beam 5 alpha 0
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam5_alpha0.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 5 \
    --mod-decode \
    --small-sample 80 \
    --alpha 0 \
    --max-len 300

# TRANSLATE beam 5 alpha 0.2
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam5_alpha0_2.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 5 \
    --mod-decode \
    --small-sample 80 \
    --alpha 0.2 \
    --max-len 300

# TRANSLATE beam 5 alpha 0.5
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam5_alpha0_5.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 5 \
    --mod-decode \
    --small-sample 80 \
    --alpha 0.5 \
    --max-len 300

# TRANSLATE beam 5 alpha 0.7
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam5_alpha0_7.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 5 \
    --mod-decode \
    --small-sample 80 \
    --alpha 0.7 \
    --max-len 300

# TRANSLATE beam 5 alpha 1
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam5_alpha1.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 5 \
    --mod-decode \
    --small-sample 80 \
    --alpha 1 \
    --max-len 300