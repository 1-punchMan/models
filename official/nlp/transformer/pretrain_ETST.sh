export PYTHONPATH="$PYTHONPATH:/home/zchen/encyclopedia-text-style-transfer/models/"
export CUDA_VISIBLE_DEVICES=1

PARAM_SET=base
WIKIPATH="/home/zchen/encyclopedia-text-style-transfer/models/official/nlp/transformer/data/ETST/wiki/"
BAIDUPATH="/home/zchen/encyclopedia-text-style-transfer/models/official/nlp/transformer/data/ETST/baidu/"
OUT_DIR=/home/zchen/encyclopedia-text-style-transfer/models/official/nlp/transformer/experiments/ETST_pretrain/1
VOCAB_FILE="/home/zchen/encyclopedia-text-style-transfer/data/vocab"

# Train the model for {train_steps} steps and evaluate every {steps_between_evals} steps on a single GPU.
# Each train step, takes {batch_size} tokens as a batch budget with {max_length} as sequence
# maximal length.
python transformer_main.py \
    --task ETST \
    --wiki_dir=$WIKIPATH --baidu_dir=$BAIDUPATH --model_dir=$OUT_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
    --train_steps=10000000 --steps_between_evals=6000 --early_stopping 25 \
    --batch_size=3000 --max_length=256 \
    --num_gpus=1 \
    --enable_time_history=false \
    --enable_tensorboard=100 \
    --enable_metrics_in_training=true