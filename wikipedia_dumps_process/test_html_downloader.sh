# Description: This script runs the pipeline for downloading Wikipedia pages

while getopts ":i:w:o:" opt;
do
    case $opt in
        i) input_file=$OPTARG ;;
        w) num_workers=$OPTARG ;;
        o) output_dir=$OPTARG ;;
        ?) echo "Invalid option: -$OPTARG"
           echo "Usage: $0 [-i <input_file>] [-w <num_workers>] [-o <output_dir>]"
           echo "Options:"
           echo "i: input file"
           echo "w: number of concurrent requests"
           echo "o: output directory"
           exit 1
           ;;
    esac
done

echo "Input file: $input_file"
echo "Number of concurrent requests: $num_workers"
echo "Output directory: $output"

# Check if all arguments are provided
if [ -z "$input_file" ] || [ -z "$num_workers" ] || [ -z "$output_dir" ]
then
    echo "Missing arguments!"
    echo "Usage: $0 [-i <input_file>] [-w <num_workers>] [-o <output_dir>]"
    echo "Options:"
    echo "i: input file"
    echo "w: number of concurrent requests"
    echo "o: output directory"
    exit 1
fi

# Run the python script
echo "Preparing data..."
python test_html_downloader_prep.py \
    --input_file $input_file \
    --output_dir $output_dir 

echo "Downloading pages..."
node crawler/crawl_wiki.js \
    --articles $output_dir"/pages.txt" \
    --concurrency $num_workers \
    --destinationDirectory $output_dir

# End of script