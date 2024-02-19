export HF_HOME=/dlabdata1/tsoares/.cache/hugging_face/ \
&& \
source /dlabdata1/tsoares/linkrec-llms/data_modelling/training_ranking/.bashrc_wendler \
&& \
conda activate /dlabdata1/tsoares/conda/envs/runai/ \
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANGUAGE \
    --models_dir /dlabdata1/tsoares/models \
    --column_name roberta_simple \
    --models_prefix roberta_simple \
    --use_cuda \
    --multilingual_name none \
&& \
python benchmark_models.py \
    --data_dir test_data_full/ \
    --langs $LANGUAGE \
    --models_dir /dlabdata1/tsoares/models \
    --column_name roberta_full \
    --models_prefix roberta_full \
    --use_section_title \
    --use_mentions \
    --multilingual_name multilingual-all \
    --only_multilingual \
    --use_cuda \
&& \
python benchmark_models.py \
    --data_dir test_data_full/ \
    --langs $LANGUAGE \
    --models_dir /dlabdata1/tsoares/models \
    --column_name roberta_full \
    --models_prefix roberta_full \
    --use_section_title \
    --use_mentions \
    --multilingual_name multilingual \
    --use_cuda \
    --only_multilingual
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANGUAGE \
    --models_dir /dlabdata1/tsoares/models \
    --column_name roberta_expansion \
    --models_prefix roberta_expansion \
    --use_cuda \
    --multilingual_name none \
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANGUAGE \
    --models_dir /dlabdata1/tsoares/models \
    --column_name roberta_dyn_mask \
    --models_prefix roberta_dyn_mask \
    --use_cuda \
    --multilingual_name none \
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANGUAGE \
    --models_dir /dlabdata1/tsoares/models \
    --column_name roberta_dyn_mask_no_neg \
    --models_prefix roberta_dyn_mask_no_neg \
    --use_cuda \
    --multilingual_name none \
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANGUAGE \
    --models_dir /dlabdata1/tsoares/models \
    --column_name roberta_only_expansion \
    --models_prefix roberta_only_expansion \
    --use_cuda \
    --use_section_title \
    --use_mentions \
    --multilingual_name none \
&& \
python benchmark_models.py \
    --data_dir test_data/ \
    --langs $LANGUAGE \
    --models_dir /dlabdata1/tsoares/models \
    --column_name roberta_pointwise \
    --models_prefix roberta_pointwise \
    --use_cuda \
    --multilingual_name none \
    --use_section_title \
    --use_mentions \
    --pointwise_loss
