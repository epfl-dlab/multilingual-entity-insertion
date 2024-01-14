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
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANG \
    --models_dir /dlabdata1/tsoares/models/multilingual/ \
    --column_name roberta_full \
    --models_prefix roberta_full \
    --use_section_title \
    --use_mentions \
    --use_cuda \
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANG \
    --models_dir /dlabdata1/tsoares/models/multilingual/ \
    --column_name roberta_simple \
    --models_prefix roberta_simple \
    --use_cuda