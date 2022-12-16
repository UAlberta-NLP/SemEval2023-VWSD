OUTPUT_FOLDER=$(realpath outputs/$(basename $0 .sh))
INPUT_PATH=outputs/03-retrieve-images/data.tsv

mkdir -p $OUTPUT_FOLDER

python3 filter_urls.py \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_FOLDER/data.tsv