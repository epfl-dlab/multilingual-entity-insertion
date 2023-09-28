# Description: This script runs the pipeline for processing wikipedia dumps.

start=`date +%s`

while getopts "i:l:c:p:o:" opt
do
    case $opt in
        i) input=$OPTARG ;;
        l) language=$OPTARG ;;
        c) chunk=$OPTARG ;;
        p) processes=$OPTARG ;;
        o) output=$OPTARG ;;
        ?) echo "Invalid option: -$OPTARG"
           echo "Usage: $0 [-i <input>] [-l <language>] [-c <chunk_size>] [-p <processes>] [-o <output>]"
           echo "Options:"
           echo "i: input directory"
           echo "l: language of the wikipedia dump"
           echo "c: chunk size"
           echo "p: number of processes"
           echo "o: output directory"
           exit 1
           ;;
    esac
done

echo "Input directory: $input"
echo "Language: $language"
echo "Chunk size: $chunk"
echo "Number of processes: $processes"
echo "Output directory: $output"

# Check if all arguments are provided
if [ -z "$input" ] || [ -z "$language" ] || [ -z "$chunk" ] || [ -z "$processes" ] || [ -z "$output" ]
then
    echo "Missing arguments!"
    echo "Usage: $0 [-i <input>] [-l <language>] [-c <chunk_size>] [-p <processes>] [-o <output>]"
    echo "Options:"
    echo "i: input directory"
    echo "l: language of the wikipedia dump"
    echo "c: chunk size"
    echo "p: number of processes"
    echo "o: output directory"
    exit 1
fi

# define directory for intermediate storage
inter=$output"_inter"

# Run the python scripts
echo "Extracting information about all the pages..."
python pages_extractor_pool.py \
    --input_dir $input \
    --language $language \
    --output_dir $inter \
    --chunksize $chunk \
    --processes $processes

echo "Extracting information about all the links..."
python link_extractor_pool.py \
    --input_dir $inter \
    --page_ids $inter"/simple_pages.parquet" \
    --redirect_map $inter"/redirect_map.parquet" \
    --output_dir $output \
    --processes $processes

# Remove the intermediate directory
rm -rf $inter

# Print the time taken
end=`date +%s`
runtime=$((end-start))
echo "Time taken: $runtime seconds"

# End of script