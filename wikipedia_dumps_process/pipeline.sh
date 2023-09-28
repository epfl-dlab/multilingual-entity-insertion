# Description: This script runs the pipeline for processing wikipedia dumps.

start=`date +%s`

while getopts "i:l:p:o:" opt
do
    case $opt in
        i) input=$OPTARG ;;
        l) language=$OPTARG ;;
        p) processes=$OPTARG ;;
        o) output=$OPTARG ;;
        ?) echo "Invalid option: -$OPTARG"
           echo "Usage: $0 [-i <input>] [-l <language>] [-p <processes>] [-o <output>]"
           echo "Options:"
           echo "i: input compressed file"
           echo "l: language of the wikipedia dump"
           echo "p: number of processes"
           echo "o: output directory"
           exit 1
           ;;
    esac
done

echo "Input directory: $input"
echo "Language: $language"
echo "Number of processes: $processes"
echo "Output directory: $output"

# Check if all arguments are provided
if [ -z "$input" ] || [ -z "$language" ] || [ -z "$processes" ] || [ -z "$output" ]
then
    echo "Missing arguments!"
    echo "Usage: $0 [-i <input>] [-l <language>] [-p <processes>] [-o <output>]"
    echo "Options:"
    echo "i: input compressed file"
    echo "l: language of the wikipedia dump"
    echo "p: number of processes"
    echo "o: output directory"
    exit 1
fi

# Run the python scripts
echo "Extracting information about all the pages..."
python pages_extractor_pool.py \
    --input_file $input \
    --language $language \
    --output_dir $output

echo "Extracting information about all the links..."
python link_extractor_pool.py \
    --input_dir $output \
    --page_ids $output"/simple_pages.parquet" \
    --redirect_map $output"/redirect_map.parquet" \
    --output_dir $output \
    --processes $processes

# Print the time taken
end=`date +%s`
runtime=$((end-start))
echo "Time taken: $runtime seconds"

# End of script