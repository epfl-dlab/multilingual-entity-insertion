export HF_HOME=/dlabdata1/tsoares/.cache/hugging_face/ \
&& \
echo "HF_HOME: $HF_HOME" > runai_log_5.txt \
&& \
echo "MODEL_NAME: $MODEL_NAME" >> runai_log_5.txt \
&& \
echo "MODEL_ARCHITECTURE: $MODEL_ARCHITECTURE" >> runai_log_5.txt \
&& \
source .bashrc_wendler \
&& \
echo "New .bashrc loaded" >> runai_log_5.txt \
&& \
conda activate /dlabdata1/tsoares/conda/envs/runai/ \
&& \
echo "Conda environment activated" >> runai_log_5.txt \
&& \
accelerate launch main_list_softmax.py \
    --model_name $MODEL_NAME \
    --model_architecture $MODEL_ARCHITECTURE \
    --data_dir datasets/simple_stage_1_large_long \
    --data_dir_2 datasets/simple_stage_2_long \
    --num_epochs 1 10 \
    --batch_size $BATCH_SIZE \
    --num_workers 52 \
    --lr 0.00001 \
    --gamma_lr 1 \
    --print_steps 500 10 \
    --save_steps 10000 500 \
    --eval_steps 5000 500 \
    --scheduler_steps 10000 500 \
    --ga_steps 2 1 \
    --full_freeze_steps 0 \
    --freeze_layers 0 \
    --head_lr_factor 20 \
    --neg_samples_train 19 \
    --neg_samples_eval 49 \
    --temperature 1 \
    --insert_section \
    --insert_mentions full \
    --mask_negatives \
    --split_models \
    --two_stage 