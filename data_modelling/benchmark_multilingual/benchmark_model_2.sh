export HF_HOME=/dlabdata1/tsoares/.cache/hugging_face/ \
&& \
echo "HF_HOME: $HF_HOME" \
&& \
source /dlabdata1/tsoares/linkrec-llms/data_modelling/training_ranking/.bashrc_wendler \
&& \
echo "New .bashrc loaded" \
&& \
conda activate /dlabdata1/tsoares/conda/envs/runai/ \
&& \
echo "Conda environment activated" \
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANG \
    --models_dir /dlabdata1/tsoares/models/multilingual/ablations/ \
    --column_name roberta_mask_rank \
    --models_prefix roberta_mask \
    --use_cuda \
    --only_multilingual \
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANG \
    --models_dir /dlabdata1/tsoares/models/multilingual/ablations/ \
    --column_name roberta_mention_rank \
    --models_prefix roberta_mention \
    --use_section_title \
    --use_mentions \
    --use_cuda \
    --only_multilingual \
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANG \
    --models_dir /dlabdata1/tsoares/models/multilingual/ablations/ \
    --column_name roberta_only_stage_two_rank \
    --models_prefix roberta_only_stage_two \
    --use_section_title \
    --use_mentions \
    --use_cuda \
    --only_multilingual \
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANG \
    --models_dir /dlabdata1/tsoares/models/multilingual/ablations/ \
    --column_name roberta_two_stage_rank \
    --models_prefix roberta_two_stage \
    --use_section_title \
    --use_mentions \
    --use_cuda \
    --only_multilingual \
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANG \
    --models_dir /dlabdata1/tsoares/models/multilingual/ablations/ \
    --column_name roberta_simple_rank \
    --models_prefix roberta_simple \
    --use_cuda \
    --only_multilingual \
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANG \
    --models_dir /dlabdata1/tsoares/models/multilingual/ablations/ \
    --column_name roberta_section_rank \
    --models_prefix roberta_section \
    --use_section_title \
    --use_cuda \
    --only_multilingual \
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANG \
    --models_dir /dlabdata1/tsoares/models/multilingual/ablations/ \
    --column_name roberta_two_stage_no_corrupt_rank \
    --models_prefix roberta_two_stage_no_corrupt \
    --use_section_title \
    --use_mentions \
    --use_cuda \
    --only_multilingual