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
python benchmark_baselines_all.py \
  --data_path test_ranking_scores_all.parquet \
  --mention_map mentions.parquet \
  --model_name BAAI/bge-base-en-v1.5 \
  --method_name embedding_similarity