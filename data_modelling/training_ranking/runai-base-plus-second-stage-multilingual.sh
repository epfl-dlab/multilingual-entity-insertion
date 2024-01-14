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
accelerate launch main_list_softmax_v2.py \
    --model_name $MODEL_NAME \
    --model_architecture $MODEL_ARCHITECTURE \
    --data_dir multilingual_datasets/stage_1/${LANGUAGE} \
    --data_dir_2 multilingual_datasets/stage_2/${LANGUAGE} \
    --num_epochs 2 2 \
    --batch_size $BATCH_SIZE \
    --num_workers 52 \
    --lr 0.000005 \
    --gamma_lr 0.95 \
    --print_steps 500 10 \
    --save_steps 10000 5000 \
    --eval_steps 5000 5000 \
    --scheduler_steps 10000 5000 \
    --ga_steps 1 1 \
    --full_freeze_steps 0 \
    --freeze_layers 0 \
    --head_lr_factor 20 \
    --neg_samples_train 9 \
    --neg_samples_eval 19 \
    --temperature 1 \
    --insert_section \
    --mask_negatives \
    --max_tokens 512 \
    --insert_mentions \
    --two_stage