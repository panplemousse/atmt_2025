#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=beam_and_optimize_assignment5.out

module load a100
module load miniforge3
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# TRANSLATE beam 1 (greedy)
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam1_ori.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 1 \
    --small-sample 80 \
    --max-len 300

# TRANSLATE beam 1 (greedy) modified decode
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam1_mod.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 1 \
    --mod-decode \
    --small-sample 80 \
    --max-len 300


# TRANSLATE beam 3
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam3_ori.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 3\
    --max-len 300

# TRANSLATE beam 3 modified decode
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam3_mod.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 3\
    --mod-decode \
    --small-sample 80 \
    --max-len 300

# TRANSLATE beam 5
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam5_ori.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 5\
    --small-sample 80 \
    --max-len 300

# TRANSLATE beam 5 modified decode
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam5_mod.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 5\
    --mod-decode \
    --small-sample 80 \
    --max-len 300