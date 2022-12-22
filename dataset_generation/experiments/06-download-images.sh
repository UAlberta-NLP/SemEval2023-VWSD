OUTPUT_FOLDER=$(realpath outputs/$(basename $0 .sh))
INPUT_PATH=outputs/05-filter-url-extensions/data.tsv
mkdir -p $OUTPUT_FOLDER

python3 download_images.py \
    --input_path $INPUT_PATH \
    --output_folder $OUTPUT_FOLDER