export HF_HOME=/dlabdata1/tsoares/.cache/hugging_face/ \
&& \
source .bashrc_wendler \
&& \
conda activate /dlabdata1/tsoares/conda/envs/runai/ \
&& \
accelerate launch main.py \
    --model_name $MODEL_NAME \
    --model_architecture $MODEL_ARCHITECTURE \
    --data_dir multilingual_datasets/stage_1/${LANGUAGE} \
    --data_dir_2 multilingual_datasets/stage_2/${LANGUAGE} \
    --num_epochs 4 0 \
    --batch_size $BATCH_SIZE \
    --num_workers 52 \
    --lr 0.000005 \
    --gamma_lr 1 \
    --print_steps 500 1000 \
    --save_steps 100000 500000 \
    --eval_steps 500000 500000 \
    --scheduler_steps 10000 50000 \
    --ga_steps 1 1 \
    --full_freeze_steps 0 \
    --freeze_layers 0 \
    --head_lr_factor 10 \
    --neg_samples_train 9 \
    --neg_samples_eval 19 \
    --temperature 1 \
    --max_tokens 512 \
    --no_mask_perc 1 \
    --mask_mention_perc 0 \
    --mask_sentence_perc 0 \
    --mask_span_perc 0
