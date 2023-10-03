# Description: This script runs the pipeline for processing wikipedia dumps.

start=`date +%s`

while getopts ":x:s:t:l:p:o:" opt;
do
    case $opt in
        x) input_xml=$OPTARG ;;
        s) input_sql=$OPTARG ;;
        t) input_tar=$OPTARG ;;
        l) language=$OPTARG ;;
        p) processes=$OPTARG ;;
        o) output=$OPTARG ;;
        ?) echo "Invalid option: -$OPTARG"
           echo "Usage: $0 [-x <input_xml>] [-s <input_sql>] [-t <input_tar>] [-l <language>] [-p <processes>] [-o <output>]"
           echo "Options:"
           echo "x: input xml file"
           echo "s: input sql file"
           echo "t: input tar file"
           echo "l: language of the wikipedia dump"
           echo "p: number of processes"
           echo "o: output directory"
           exit 1
           ;;
    esac
done

echo "Input xml file: $input_xml"
echo "Input sql file: $input_sql"
echo "Input tar file: $input_tar"
echo "Language: $language"
echo "Number of processes: $processes"
echo "Output directory: $output"

# Check if all arguments are provided
if [ -z "$input_xml" ] || [ -z "$input_sql" ] || [ -z "$input_tar" ] || [ -z "$language" ] || [ -z "$processes" ] || [ -z "$output" ]
then
    echo "Missing arguments!"
    echo "Usage: $0 [-x <input_xml>] [-s <input_sql>] [-t <input_tar>] [-l <language>] [-p <processes>] [-o <output>]"
    echo "Options:"
    echo "x: input xml file"
    echo "s: input sql file"
    echo "t: input tar file"
    echo "l: language of the wikipedia dump"
    echo "p: number of processes"
    echo "o: output directory"
    exit 1
fi

# Run the python scripts
echo "Extracting information about all the pages..."
python pages_extractor.py \
    --input_xml $input_xml \
    --input_sql $input_sql \
    --input_tar $input_tar \
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