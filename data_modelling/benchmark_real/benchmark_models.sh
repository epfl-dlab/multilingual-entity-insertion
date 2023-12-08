export HF_HOME=/dlabdata1/tsoares/.cache/hugging_face/ \
&& \
source /dlabdata1/tsoares/linkrec-llms/data_modelling/training_ranking/.bashrc_wendler \
&& \
conda activate /dlabdata1/tsoares/conda/envs/runai/ \
&& \
python benchmark_models_all.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name bert_two_stage_rank \
  --model_name bert_two_stage \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name roberta_simple \
  --model_name roberta_simple \
  --batch_size 36 \
  --loss_function ranking \
&& \
python benchmark_models_all.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name roberta_mask \
  --model_name roberta_mask \
  --batch_size 36 \
  --loss_function ranking \
&& \
python benchmark_models_all.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name roberta_mask_section \
  --model_name roberta_section \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
&& \
python benchmark_models_all.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name roberta_mask_section_mention \
  --model_name roberta_mention \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name roberta_two_stage \
  --model_name roberta_mention \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions \
&& \
python benchmark_models_all.py \
  --data_path test_ranking_scores_all.parquet \
  --models_dir /dlabdata1/tsoares/models \
  --mention_map mentions.parquet \
  --column_name roberta_only_two_stage \
  --model_name roberta_only_two_stage \
  --batch_size 36 \
  --loss_function ranking \
  --use_section_title \
  --use_mentions














python benchmark_models.py \
  --data_path test_ranking_scores.parquet \
  --models_dir /scratch/tsoares/models \
  --mention_map mentions.parquet \
  --column_name model_${LOSS}_corrupt_section_random_rank \
  --batch_size 36 \
  --loss_function ${LOSS} \
  --use_section_title_random \
  --use_corruption \
&& \
python benchmark_models.py \
  --data_path test_ranking_scores.parquet \
  --models_dir /scratch/tsoares/models \
  --mention_map mentions.parquet \
  --column_name model_${LOSS}_corrupt_section_rank \
  --batch_size 36 \
  --loss_function ${LOSS} \
  --use_section_title \
  --use_corruption \
&& \
python benchmark_models.py \
  --data_path test_ranking_scores.parquet \
  --models_dir /scratch/tsoares/models \
  --mention_map mentions.parquet \
  --column_name model_${LOSS}_corrupt_rank \
  --batch_size 36 \
  --loss_function ${LOSS} \
  --use_corruption \
&& \
python benchmark_models.py \
  --data_path test_ranking_scores.parquet \
  --models_dir /scratch/tsoares/models \
  --mention_map mentions.parquet \
  --column_name model_${LOSS}_rank \
  --batch_size 36 \
  --loss_function ${LOSS} \
&& \
python benchmark_models.py \
  --data_path test_ranking_scores.parquet \
  --models_dir /scratch/tsoares/models \
  --mention_map mentions.parquet \
  --column_name model_${LOSS}_corrupt_section_mentions_negmask_rank \
  --batch_size 36 \
  --loss_function ${LOSS} \
  --use_section_title \
  --use_mentions \
  --use_corruption \
  --mask_negatives