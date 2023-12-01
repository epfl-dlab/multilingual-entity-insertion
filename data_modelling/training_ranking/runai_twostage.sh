export HF_HOME=/dlabdata1/tsoares/.cache/hugging_face/ \
&& \
echo "HF_HOME: $HF_HOME" > runai_log.txt \
&& \
source .bashrc_wendler \
&& \
echo "New .bashrc loaded" >> runai_log.txt \
&& \
conda activate /dlabdata1/tsoares/conda/envs/runai/ \
&& \
echo "Conda environment activated" >> runai_log.txt \
&& \
accelerate launch main_list_softmax.py \
    --model_name bert-base-uncased \
    --data_dir datasets/simple_stage_1 \
    --data_dir_2 datasets/simple_stage_2 \
    --num_epochs 2 10 \
    --batch_size 2 \
    --num_workers 52 \
    --lr 0.00002 \
    --gamma_lr 0.9 \
    --print_steps 500 10 \
    --save_steps 10000 1000 \
    --eval_steps 5000 500 \
    --scheduler_steps 5000 500 \
    --ga_steps 2 2 \
    --full_freeze_steps 0 \
    --freeze_layers 2 \
    --head_lr_factor 5 \
    --neg_samples_train 9 \
    --neg_samples_eval 19 \
    --temperature 1 \
    --insert_mentions candidates \
    --insert_section \
    --mask_negatives \
    --two_stage \
    --current_links_mode weighted_average \
    --normalize_current_links \
    --n_links 9 \
    --delay_fuser_steps 10000
