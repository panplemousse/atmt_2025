#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out4_assignment3.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit


# PREPARE DATA
python preprocess.py \
    --source-lang cz \
    --target-lang en \
    --raw-data ~/shares/cz-en/data/raw \
    --dest-dir ./cz-en/data/prepared \
    --model-dir ./cz-en/tokenizers \
    --test-prefix test \
    --train-prefix train \
    --valid-prefix valid \
    --src-vocab-size 8000 \
    --tgt-vocab-size 8000 \
    --src-model ./cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-model ./cz-en/tokenizers/en-bpe-8000.model

# TRAIN
#python train.py \
#    --cuda \
#    --data cz-en/data/prepared/ \
#    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
#    --source-lang cz \
#    --target-lang en \
#    --batch-size 64 \
#    --arch gqa-transformer \
#    --query-groups 2 \
#    --max-epoch 4 \
#    --log-file cz-en/logs/gqa/train.log \
#    --save-dir cz-en/checkpoints/gqa/ \
#    --epoch-checkpoints \
#    --ignore-checkpoints \
#    --encoder-dropout 0.1 \
#    --decoder-dropout 0.1 \
#    --dim-embedding 256 \
#    --attention-heads 4 \
#    --dim-feedforward-encoder 1024 \
#    --dim-feedforward-decoder 1024 \
#    --max-seq-len 300 \
#    --n-encoder-layers 3 \
#    --n-decoder-layers 3 


# REFINE
python train.py \
    --cuda \
    --data cz-en/data/prepared/ \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --source-lang cz \
    --target-lang en \
    --batch-size 64 \
    --arch gqa-transformer \
    --query-groups 2 \
    --max-epoch 7 \
    --log-file cz-en/logs/gqa/refine.log \
    --save-dir cz-en/checkpoints/gqa/ \
    --restore-file checkpoint_last.pt \
    --epoch-checkpoints \
    --average-n-checkpoints 3 \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 300 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3 

# TRANSLATE-Best Model
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_best.pt \
    --output cz-en/outputs/gqa/output.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --max-len 300

# TRANSLATE-Averaged Model
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/gqa/checkpoint_averaged_last3.pt \
    --output cz-en/outputs/gqa/output_refine.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --max-len 300
