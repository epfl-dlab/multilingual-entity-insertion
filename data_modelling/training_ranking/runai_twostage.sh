echo "Running on host" > /dlabdata1/tsoares/linkrec-llms/data_modelling/training_ranking/hostname.txt \
&& \
export HF_HOME=/dlabdata1/tsoares/.cache/hugging_face/ \
&& \
echo "HF_HOME set" >> /dlabdata1/tsoares/linkrec-llms/data_modelling/training_ranking/hostname.txt \
&& \
conda activate /dlabdata1/tsoares/conda/envs/runai/ \
&& \
echo "Conda environment activated" >> /dlabdata1/tsoares/linkrec-llms/data_modelling/training_ranking/hostname.txt \
&& \
cd /dlabdata1/tsoares/linkrec-llms/data_modelling/training_ranking/ \
&& \
ls >> /dlabdata1/tsoares/linkrec-llms/data_modelling/training_ranking/hostname.txt \
&& \
echo "Changed directory" >> /dlabdata1/tsoares/linkrec-llms/data_modelling/training_ranking/hostname.txt \
&& \
accelerate launch main_list_softmax.py --model_name bert-base-uncased --data_dir datasets/simple_stage_1 --data_dir_2 datasets/simple_stage_2 --num_epochs 2 10 --batch_size 20 --num_workers 52 --lr 0.00001 0.00001 --gamma_lr 0.9 0.9 --print_steps 100 10 --save_steps 1000 250 --eval_steps 1000 250 --scheduler_steps 1000 250 --ga_steps 1 1 --full_freeze_epochs 0 --freeze_layers 2 --head_lr_factor 20 --neg_samples_train 9 9 --neg_samples_eval 19 19 --temperature 1 1 --insert_mentions candidates --insert_section --mask_negatives --two_stage \
&& \
echo "Finished running" >> /dlabdata1/tsoares/linkrec-llms/data_modelling/training_ranking/hostname.txt
