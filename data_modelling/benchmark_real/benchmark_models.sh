export HF_HOME=/dlabdata1/tsoares/.cache/hugging_face/ \
&& \
echo "HF_HOME: $HF_HOME" > log.txt \
&& \
source /dlabdata1/tsoares/linkrec-llms/data_modelling/training_ranking/.bashrc_wendler \
&& \
echo "New .bashrc loaded" >> log.txt \
&& \
conda activate /dlabdata1/tsoares/conda/envs/runai/ \
&& \
echo "Conda environment activated" >> log.txt \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name bert_two_stage_v2_rank \
  --model_name bert_two_stage_v2 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --column_name roberta_simple_v2_rank \
  --model_name roberta_simple_v2 \
  --batch_size 36 \
  --loss_function ranking \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --column_name roberta_mask_v2_rank \
  --model_name roberta_mask_v2 \
  --batch_size 36 \
  --loss_function ranking \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --column_name roberta_mask_section_v2_rank \
  --model_name roberta_section_v2 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name roberta_mask_section_mention_v2_rank \
  --model_name roberta_mention_v2 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --mention_map mentions.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --column_name roberta_two_stage_v2_rank \
  --model_name roberta_two_stage_v2 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name roberta_only_stage_two_v2_rank \
  --model_name roberta_only_stage_two_v2 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name roberta_two_stage_no_corrupt_v2_rank \
  --model_name roberta_two_stage_no_corrupt_v2 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name t5_two_stage_v2_rank \
  --model_name t5_two_stage_v2 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
  --encoder_decoder \ 
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name roberta_large_two_stage_rank \
  --model_name roberta_large_two_stage \
  --batch_size 24 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models/roberta_input_size \
  --mention_map mentions.parquet \
  --column_name roberta_stage_2_100 \
  --model_name stage_2_100 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models/roberta_input_size \
  --mention_map mentions.parquet \
  --column_name roberta_stage_2_1000 \
  --model_name stage_2_1000 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models/roberta_input_size \
  --mention_map mentions.parquet \
  --column_name roberta_stage_2_10000 \
  --model_name stage_2_10000 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models/roberta_input_size \
  --mention_map mentions.parquet \
  --column_name roberta_stage_2_100000 \
  --model_name stage_2_100000 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models/roberta_input_size \
  --mention_map mentions.parquet \
  --column_name roberta_stage_1_1000 \
  --model_name stage_1_1000 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models/roberta_input_size \
  --mention_map mentions.parquet \
  --column_name roberta_stage_1_10000 \
  --model_name stage_1_10000 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models/roberta_input_size \
  --mention_map mentions.parquet \
  --column_name roberta_stage_1_100000 \
  --model_name stage_1_100000 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all_v2.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models/roberta_input_size \
  --mention_map mentions.parquet \
  --column_name roberta_stage_1_1000000 \
  --model_name stage_1_1000000 \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions