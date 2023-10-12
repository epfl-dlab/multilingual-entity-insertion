# Description: This script runs the pipeline for processing wikipedia dumps.

start=`date +%s`

while getopts ":i:l:d:p:o:" opt;
do
    case $opt in
        i) input_dir=$OPTARG ;;
        l) language=$OPTARG ;;
        d) date=$OPTARG ;;
        p) processes=$OPTARG ;;
        o) output=$OPTARG ;;
        ?) echo "Invalid option: -$OPTARG"
           echo "Usage: $0 [-i <input_dir>] [-l <language>] [-d <date>] [-p <processes>] [-o <output>]"
           echo "Options:"
           echo "i: input directory"
           echo "l: language of the wikipedia dump"
           echo "d: date of the wikipedia dump"
           echo "p: number of processes"
           echo "o: output directory"
           exit 1
           ;;
    esac
done

echo "Input directory: $input_dir"
echo "Language: $language"
echo "Date: $date"
echo "Number of processes: $processes"
echo "Output directory: $output"

# Check if all arguments are provided
if [ -z "$input_dir" ] || [ -z "$language" ] || [ -z "$date" ] || [ -z "$processes" ] || [ -z "$output" ]
then
    echo "Missing arguments!"
    echo "Usage: $0 [-i <input_dir>] [-l <language>] [-d <date>] [-p <processes>] [-o <output>]"
    echo "Options:"
    echo "i: input directory"
    echo "l: language of the wikipedia dump"
    echo "d: date of the wikipedia dump"
    echo "p: number of processes"
    echo "o: output directory"
    exit 1
fi

# Run the python scripts
echo "Extracting information about all the pages..."
python pages_extractor.py \
    --input_dir $input_dir \
    --language $language \
    --date $date \
    --output_dir $output

echo "Extracting information about all the links..."
python link_extractor_pool.py \
    --input_dir $output \
    --page_ids $output"/simple_pages.parquet" \
    --redirect_map $output"/redirect_map.parquet" \
    --output_dir $output \
    --processes $processes

echo "Running the post-processing script to clean-up the data..."
python post_processing.py \
    --input_dir $output \
    --output_dir $output

# Print the time taken
end=`date +%s`
runtime=$((end-start))
echo "Time taken: $runtime seconds"

# End of script