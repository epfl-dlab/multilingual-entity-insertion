for FREEZE_LAYERS in 0 1 2 4 6
do
  accelerate launch main.py \
    --model_name bert-base-uncased \
    --data_dir /scratch/tsoares/wikidumps/simplewiki-NS0-20231001/ml_data \
    --num_epochs 1 \
    --batch_size 4 \
    --num_workers 16 \
    --lr 0.00001 \
    --gamma_lr 0.9 \
    --print_steps 50 \
    --save_steps 2000 \
    --eval_steps 500 \
    --scheduler_steps 3000 \
    --ga_steps 2 \
    --full_freeze_epochs 0 \
    --freeze_layers $FREEZE_LAYERS \
    --head_lr_factor 50
done