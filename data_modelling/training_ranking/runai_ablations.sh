for TEMP in 0.1 1 10 50 100
do
  for LR in 0.0001 0.00001 0.000001
  do
    for HEAD_FACTOR in 1 20 50
    do
      accelerate launch main_list_softmax.py \
        --model_name bert-base-uncased \
        --data_dir datasets/ml_data_synth_large_hard \
        --num_epochs 2 \
        --batch_size 20 \
        --num_workers 52 \
        --lr $LR \
        --gamma_lr 0.9 \
        --print_steps 100 \
        --save_steps 1000 \
        --eval_steps 1000 \
        --scheduler_steps 1000 \
        --ga_steps 1 \
        --full_freeze_epochs 0 \
        --freeze_layers 2 \
        --head_lr_factor $HEAD_FACTOR \
        --neg_samples_train 9 \
        --neg_samples_eval 19 \
        --temperature $TEMP \
        --insert_mentions candidates \
        --insert_section \
        --mask_negatives
    done
  done
done