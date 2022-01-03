export PYTHONPATH="$PYTHONPATH:/home/zchen/encyclopedia-text-style-transfer/models/"
export CUDA_VISIBLE_DEVICES=1

PARAM_SET=base
DATA_DIR="/home/zchen/encyclopedia-text-style-transfer/models/official/nlp/transformer/data/en-de_data/"
MODEL_DIR=./model_$PARAM_SET/2
VOCAB_FILE=$DATA_DIR/vocab.ende.32768

# Train the model for 100000 steps and evaluate every 5000 steps on a single GPU.
# Each train step, takes 4096 tokens as a batch budget with 64 as sequence
# maximal length.
python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
    --train_steps=100000 --steps_between_evals=50 \
    --batch_size=4096 --max_length=64 \
    `# --bleu_source=$DATA_DIR/newstest2014.en \
    # --bleu_ref=$DATA_DIR/newstest2014.de` \
    --num_gpus=1 \
    --enable_time_history=false \
    --enable_tensorboard=true