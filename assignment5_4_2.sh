#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=stopping_criteria_assignment5.out

module load a100
module load miniforge3
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit


# TRANSLATE beam 5 alpha 0.2 constant
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam5_alpha0_2_constant.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 5 \
    --mod-decode \
    --small-sample 80 \
    --alpha 0.2 \
    --max-len 300 \

# TRANSLATE beam 5 alpha 0.2 local threshold
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam5_alpha0_2_local_threshold.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 5 \
    --mod-decode \
    --small-sample 80 \
    --alpha 0.2 \
    --max-len 300 \
    --local-threshold 0.5 \


# TRANSLATE beam 5 alpha 0.2 maximum candidates
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/beam/output_beam5_alpha0_2_maximum_candidates.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --beam-size 5 \
    --mod-decode \
    --small-sample 80 \
    --alpha 0.2 \
    --max-len 300 \
    --maximum-candidates 3 \