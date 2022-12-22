OUTPUT_FOLDER=$(realpath outputs/$(basename $0 .sh))

DATASET_FOLDER=outputs/01-extract-noun-phrases
IMAGE_FOLDER=outputs/06-download-images
CANDIDATES_PATH=outputs/03-retrieve-synset-ids/data.tsv

mkdir -p $OUTPUT_FOLDER

python3 build_dataset.py \
    --dataset_folder $DATASET_FOLDER \
    --image_folder $IMAGE_FOLDER \
    --candidates_path $CANDIDATES_PATH \
    --output_folder $OUTPUT_FOLDER