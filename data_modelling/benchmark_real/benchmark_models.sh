export HF_HOME=/dlabdata1/tsoares/.cache/hugging_face/ \
&& \
source /dlabdata1/tsoares/linkrec-llms/data_modelling/training_ranking/.bashrc_wendler \
&& \
conda activate /dlabdata1/tsoares/conda/envs/runai/ \
&& \
# python benchmark_models_all.py \
#   --data_path test_ranking_scores_all.parquet \
#   --models_dir /dlabdata1/tsoares/models \
#   --mention_map mentions.parquet \
#   --column_name bert_two_stage_rank \
#   --model_name bert_two_stage \
#   --batch_size 36 \
#   --loss_function ranking \
#   --use_section_title \
#   --use_mentions \
#   --split_models \
# && \
# python benchmark_models_all.py \
#   --data_path test_ranking_scores_all.parquet \
#   --models_dir /dlabdata1/tsoares/models \
#   --mention_map mentions.parquet \
#   --column_name roberta_simple_rank \
#   --model_name roberta_simple \
#   --batch_size 36 \
#   --loss_function ranking \
#   --split_models \
# && \
# python benchmark_models_all.py \
#   --data_path test_ranking_scores_all.parquet \
#   --models_dir /dlabdata1/tsoares/models \
#   --mention_map mentions.parquet \
#   --column_name roberta_mask_rank \
#   --model_name roberta_mask \
#   --batch_size 36 \
#   --loss_function ranking \
#   --split_models \
# && \
# python benchmark_models_all.py \
#   --data_path test_ranking_scores_all.parquet \
#   --models_dir /dlabdata1/tsoares/models \
#   --mention_map mentions.parquet \
#   --column_name roberta_mask_section_rank \
#   --model_name roberta_section \
#   --batch_size 36 \
#   --loss_function ranking \
#   --use_section_title \
#   --split_models \
# && \
# python benchmark_models_all.py \
#   --data_path test_ranking_scores_all.parquet \
#   --models_dir /dlabdata1/tsoares/models \
#   --mention_map mentions.parquet \
#   --column_name roberta_mask_section_mention_rank \
#   --model_name roberta_mention \
#   --batch_size 36 \
#   --loss_function ranking \
#   --use_section_title \
#   --use_mentions \
#   --split_models \
# && \
python benchmark_models_all.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name roberta_two_stage_rank \
  --model_name roberta_two_stage \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
  --split_models \
# && \
# python benchmark_models_all.py \
#   --data_path test_ranking_scores_all.parquet \
#   --models_dir /dlabdata1/tsoares/models \
#   --mention_map mentions.parquet \
#   --column_name roberta_only_stage_two_rank \
#   --model_name roberta_only_stage_two \
#   --batch_size 36 \
#   --loss_function ranking \
#   --use_section_title \
#   --use_mentions \
#   --split_models