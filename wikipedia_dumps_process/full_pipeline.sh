# start=`date +%s`

# use getopt to parse long arguments

# VALID_ARGS=$(getopt --long first_month:,second_month:,third_month:,lang:,data_dir:,processes:,no_mask_perc:,mask_mention_perc:,mask_sentence_perc:,mask_paragraph_perc:,download_processes:,neg_samples_train:,neg_samples_val:,max_train_samples:,max_val_samples: -- "$@")
# if [[ $? -ne 0 ]]; then
#     exit 1;
# fi

# eval set -- "$VALID_ARGS"

# set default values
processes=15
no_mask_perc=0.4
mask_mention_perc=0.2
mask_sentence_perc=0.3
mask_paragraph_perc=0.1
download_processes=5
while [ : ]; do
    case "$1" in
        --first_month) first_month=$2; shift 2;;
        --second_month) second_month=$2; shift 2;;
        --third_month) third_month=$2; shift 2;;
        --lang) lang=$2; shift 2;;
        --data_dir) data_dir=$2; shift 2;;
        --processes) processes=$2; shift 2;;
        --no_mask_perc) no_mask_perc=$2; shift 2;;
        --mask_mention_perc) mask_mention_perc=$2; shift 2;;
        --mask_sentence_perc) mask_sentence_perc=$2; shift 2;;
        --mask_paragraph_perc) mask_paragraph_perc=$2; shift 2;;
        --download_processes) download_processes=$2; shift 2;;
        --neg_samples_train) neg_samples_train=$2; shift 2;;
        --neg_samples_val) neg_samples_val=$2; shift 2;;
        --max_train_samples) max_train_samples=$2; shift 2;;
        --max_val_samples) max_val_samples=$2; shift 2;;
        --) shift; break;;
        *) break;;
    esac
done

# check if any argument is missing
if [[ -z $first_month || -z $second_month || -z $third_month || -z $lang || -z $data_dir || -z $processes || -z $no_mask_perc || -z $mask_mention_perc || -z $mask_sentence_perc || -z $mask_paragraph_perc || -z $download_processes || -z $neg_samples_train || -z $neg_samples_val || -z $max_train_samples || -z $max_val_samples ]]; then
    echo "Missing arguments!"
    echo "Usage: $0 --first_month <first_month> --second_month <second_month> --third_month <third_month> --lang <lang> --data_dir <data_dir> --processes <processes> --no_mask_perc <no_mask_perc> --mask_mention_perc <mask_mention_perc> --mask_sentence_perc <mask_sentence_perc> --mask_paragraph_perc <mask_paragraph_perc> --download_processes <download_processes> --neg_samples_train <neg_samples_train> --neg_samples_val <neg_samples_val> --max_train_samples <max_train_samples> --max_val_samples <max_val_samples>"
    echo "Options:"
    echo "first_month: First month of the wikipedia dumps in format YYYYMMDD"
    echo "second_month: Second month of the wikipedia dumps in format YYYYMMDD"
    echo "third_month: Third month of the wikipedia dumps in format YYYYMMDD"
    echo "lang: Language of the wikipedia dump"
    echo "data_dir: Directory to store the data"
    echo "processes: Number of processes to use when using multiprocessing"
    echo "no_mask_perc: Percentage of links where no masking is applied"
    echo "mask_mention_perc: Percentage of links where mention masking is applied"
    echo "mask_sentence_perc: Percentage of links where sentence masking is applied"
    echo "mask_paragraph_perc: Percentage of links where paragraph masking is applied"
    echo "download_processes: Number of processes to use for downloads"
    echo "neg_samples_train: Number of negative samples for training (1st and 2nd stages)"
    echo "neg_samples_val: number of negative samples for validation (1st and 2nd stages)"
    echo "max_train_samples: Maximum number of training samples (1st stage)"
    echo "max_val_samples: Maximum number of validation samples (2nd stage)"
    exit 1
fi

# print the options to the user
echo "First Month: $first_month"
echo "Second Month: $second_month"
echo "Third Month: $third_month"
echo "Language: $lang"
echo "Data Directory: $data_dir"
echo "Number of Processes: $processes"
echo "No Mask Percentage: $no_mask_perc"
echo "Mask Mention Percentage: $mask_mention_perc"
echo "Mask Sentence Percentage: $mask_sentence_perc"
echo "Mask Paragraph Percentage: $mask_paragraph_perc"
echo "Download Processes: $download_processes"
echo "Negative Samples Train: $neg_samples_train"
echo "Negative Samples Val: $neg_samples_val"
echo "Max Train Samples: $max_train_samples"
echo "Max Val Samples: $max_val_samples"


echo "Extracting information about all the pages for the three months..."
for MONTH in $first_month $second_month $third_month; do
    echo "Extracting information about all the pages for $MONTH..."
    python pages_extractor.py \
        --input_dir ${data_dir}/${lang}wiki-NS0-${MONTH}/raw_data \
        --language $lang \
        --date $MONTH \
        --output_dir ${data_dir}/${lang}wiki-NS0-${MONTH}/processed_data
done

echo "Extracting information about all the links for the three months..."
for MONTH in $first_month $second_month $third_month; do
    echo "Extracting information about all the links for $MONTH..."
    python link_extractor_pool.py \
        --input_dir ${data_dir}/${lang}wiki-NS0-${MONTH}/processed_data \
        --page_ids ${data_dir}/${lang}wiki-NS0-${MONTH}/processed_data/simple_pages.parquet \
        --redirect_map ${data_dir}/${lang}wiki-NS0-${MONTH}/processed_data/redirect_map.parquet \
        --output_dir ${data_dir}/${lang}wiki-NS0-${MONTH}/processed_data \
        --processes $processes
done

echo "Running the post-processing script to clean-up the data for the three months..."
for MONTH in $first_month $second_month $third_month; do
    echo "Running the post-processing script to clean-up the data for $MONTH..."
    python post_processing.py \
        --input_dir ${data_dir}/${lang}wiki-NS0-${MONTH}/processed_data \
        --output_dir ${data_dir}/${lang}wiki-NS0-${MONTH}/processed_data
done

echo "Generating the synthetic test data..."
python synth_test_data_generator.py \
    --first_month_dir ${data_dir}/${lang}wiki-NS0-${second_month}/processed_data \
    --second_month_dir ${data_dir}/${lang}wiki-NS0-${third_month}/processed_data \
    --output_dir ${data_dir}/${lang}wiki-NS0-${second_month}/eval_synth \
    --no_mask_perc $no_mask_perc \
    --mask_mention_perc $mask_mention_perc \
    --mask_sentence_perc $mask_sentence_perc \
    --mask_paragraph_perc $mask_paragraph_perc

echo "Running the real test data versions finder..."
echo "Running it for the time span between $first_month and $second_month..."
python test_data_version_finder.py \
    --raw_data_dir ${data_dir}/${lang}wiki-NS0-${second_month}/raw_data \
    --first_month_dir ${data_dir}/${lang}wiki-NS0-${first_month}/processed_data \
    --second_month_dir ${data_dir}/${lang}wiki-NS0-${second_month}/processed_data \
    --output_dir ${data_dir}/${lang}wiki-NS0-${first_month}/eval \
    --lang $lang \
    --first_date $first_month \
    --second_date $second_month
echo "Running it for the time span between $second_month and $third_month..."
python test_data_version_finder.py \
    --raw_data_dir ${data_dir}/${lang}wiki-NS0-${third_month}/raw_data \
    --first_month_dir ${data_dir}/${lang}wiki-NS0-${second_month}/processed_data \
    --second_month_dir ${data_dir}/${lang}wiki-NS0-${third_month}/processed_data \
    --output_dir ${data_dir}/${lang}wiki-NS0-${second_month}/eval \
    --lang $lang \
    --first_date $second_month \
    --second_date $third_month

echo "Running HTML download prep script..."
echo "Running it for the time span between $first_month and $second_month..."
python test_html_downloader_prep.py \
    --input_file ${data_dir}/${lang}wiki-NS0-${first_month}/eval/link_versions.parquet \
    --output_directory ${data_dir}/${lang}wiki-NS0-${first_month}/eval/pages
echo "Running it for the time span between $second_month and $third_month..."
python test_html_downloader_prep.py \
    --input_file ${data_dir}/${lang}wiki-NS0-${second_month}/eval/link_versions.parquet \
    --output_directory ${data_dir}/${lang}wiki-NS0-${second_month}/eval/pages

echo "Downloading the HTML pages..."
echo "Downloading it for the time span between $first_month and $second_month..."
node crawler/crawl_wiki.js \
    --articles ${data_dir}/${lang}wiki-NS0-${first_month}/eval/pages/pages.txt \
    --concurrence $download_processes \
    --destinationDirectory ${data_dir}/${lang}wiki-NS0-${first_month}/eval/pages/ \
    --language $lang
echo "Downloading it for the time span between $second_month and $third_month..."
node crawler/crawl_wiki.js \
    --articles ${data_dir}/${lang}wiki-NS0-${second_month}/eval/pages/pages.txt \
    --concurrence $download_processes \
    --destinationDirectory ${data_dir}/${lang}wiki-NS0-${second_month}/eval/pages/ \
    --language $lang

echo "Running the test data generator..."
echo "Running it for the time span between $first_month and $second_month..."
python test_data_generator.py \
    --versions_file ${data_dir}/${lang}wiki-NS0-${first_month}/eval/link_versions.parquet \
    --html_dir ${data_dir}/${lang}wiki-NS0-${first_month}/eval/pages \
    --redirect_1 ${data_dir}/${lang}wiki-NS0-${first_month}/processed_data/redirect_map.parquet \
    --redirect_2 ${data_dir}/${lang}wiki-NS0-${second_month}/processed_data/redirect_map.parquet \
    --mention_map ${data_dir}/${lang}wiki-NS0-${first_month}/processed_data/mention_map.parquet \
    --output_dir ${data_dir}/${lang}wiki-NS0-${first_month}/eval \
    --n_processes $processes
echo "Running it for the time span between $second_month and $third_month..."
python test_data_generator.py \
    --versions_file ${data_dir}/${lang}wiki-NS0-${second_month}/eval/link_versions.parquet \
    --html_dir ${data_dir}/${lang}wiki-NS0-${second_month}/eval/pages \
    --redirect_1 ${data_dir}/${lang}wiki-NS0-${second_month}/processed_data/redirect_map.parquet \
    --redirect_2 ${data_dir}/${lang}wiki-NS0-${third_month}/processed_data/redirect_map.parquet \
    --mention_map ${data_dir}/${lang}wiki-NS0-${second_month}/processed_data/mention_map.parquet \
    --output_dir ${data_dir}/${lang}wiki-NS0-${second_month}/eval \
    --n_processes $processes

echo "Running the input generator for stage 1..."
python input_generator_stage1.py \
    --input_month1_dir ${data_dir}/${lang}wiki-NS0-${second_month}/processed_data \
    --input_month2_dir ${data_dir}/${lang}wiki-NS0-${third_month}/processed_data \
    --input_dir_val ${data_dir}/${lang}wiki-NS0-${second_month}/eval_synth \
    --output_dir ${data_dir}/${lang}wiki-NS0-${second_month}/ml_data/${lang}_stage_1 \
    --neg_strategies 6 \
    --neg_samples_train $neg_samples_train \
    --neg_samples_val $neg_samples_val \
    --max_train_samples $max_train_samples \
    --max_val_samples $max_val_samples \
    --join_samples

echo "Running the input generator for stage 2..."
python input_generator_stage2.py \
    --input_month1_dir ${data_dir}/${lang}wiki-NS0-${first_month}/processed_data \
    --input_month2_dir ${data_dir}/${lang}wiki-NS0-${second_month}/processed_data \
    --links_file ${data_dir}/${lang}wiki-NS0-${first_month}/eval/test_data.parquet \
    --output_dir ${data_dir}/${lang}wiki-NS0-${second_month}/ml_data/${lang}_stage_2 \
    --neg_samples_train $neg_samples_train \
    --neg_samples_val $neg_samples_val

# Print the time taken
# end=`date +%s`
# runtime=$((end-start))
# echo "Time taken: $runtime seconds"

# End of script