for MENTION_STRATEGY in "none"
do
  accelerate launch main_list_softmax.py \
    --model_name bert-base-uncased \
    --data_dir /scratch/tsoares/wikidumps/simplewiki-NS0-20230901/ml_data_synth_large_hard \
    --num_epochs 2 \
    --batch_size 3 \
    --num_workers 32 \
    --lr 0.00001 \
    --gamma_lr 0.9 \
    --print_steps 250 \
    --save_steps 5000 \
    --eval_steps 4000 \
    --scheduler_steps 5000 \
    --ga_steps 1 \
    --full_freeze_epochs 0 \
    --freeze_layers 2 \
    --head_lr_factor 50 \
    --neg_samples_train 9 \
    --neg_samples_eval 19 \
    --temperature 1 \
    --insert_mentions $MENTION_STRATEGY
done \
&& \
for TEMPERATURE in 0.1 10 50 100
do
  accelerate launch main_list_softmax.py \
    --model_name bert-base-uncased \
    --data_dir /scratch/tsoares/wikidumps/simplewiki-NS0-20230901/ml_data_synth_large_hard \
    --num_epochs 2 \
    --batch_size 3 \
    --num_workers 32 \
    --lr 0.00001 \
    --gamma_lr 0.9 \
    --print_steps 250 \
    --save_steps 5000 \
    --eval_steps 4000 \
    --scheduler_steps 5000 \
    --ga_steps 1 \
    --full_freeze_epochs 0 \
    --freeze_layers 2 \
    --head_lr_factor 50 \
    --neg_samples_train 9 \
    --neg_samples_eval 19 \
    --temperature $TEMPERATURE \
    --insert_mentions candidates
done
