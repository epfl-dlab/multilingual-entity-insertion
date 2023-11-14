for LOSS in 'ranking' 'indep'
do
  python benchmark_models.py \
    --data_path test_ranking_scores.parquet \
    --models_dir /scratch/tsoares/models \
    --mention_map mentions.parquet \
    --column_name model_${LOSS}_corrupt_section_mentions_rank \
    --batch_size 36 \
    --loss_function ${LOSS} \
    --use_section_title \
    --use_mentions \
    --use_corruption \
  && \
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
done