for LOSS in 'indep' 'ranking'
do
  python benchmark_models.py \
    --data_path test_ranking_scores.parquet \
    --models_dir ../../models \
    --mention_map mentions.parquet \
    --column_name model_${LOSS}_rank \
    --loss_function ${LOSS} \
  && \
  python benchmark_models.py \
    --data_path test_ranking_scores.parquet \
    --models_dir ../../models \
    --mention_map mentions.parquet \
    --column_name model_${LOSS}_corrupt_rank \
    --loss_function ${LOSS} \
    --use_corruption \
  && \
  python benchmark_models.py \
    --data_path test_ranking_scores.parquet \
    --models_dir ../../models \
    --mention_map mentions.parquet \
    --column_name model_${LOSS}_corrupt_section_rank \
    --loss_function ${LOSS} \
    --use_section_title \
    --use_corruption \
  && \
  python benchmark_models.py \
    --data_path test_ranking_scores.parquet \
    --models_dir ../../models \
    --mention_map mentions.parquet \
    --column_name model_${LOSS}_corrupt_section_random_rank \
    --loss_function ${LOSS} \
    --use_section_title_random \
    --use_corruption \
  && \
  python benchmark_models.py \
    --data_path test_ranking_scores.parquet \
    --models_dir ../../models \
    --mention_map mentions.parquet \
    --column_name model_${LOSS}_corrupt_section_mentions_rank \
    --loss_function ${LOSS} \
    --use_section_title \
    --use_mentions \
    --use_corruption
done