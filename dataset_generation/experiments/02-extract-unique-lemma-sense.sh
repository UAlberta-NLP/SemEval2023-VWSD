OUTPUT_FOLDER=$(realpath outputs/$(basename $0 .sh))
DATASET_NAMES=(dev-en test-en semcor_en wngt_examples_en wngt_glosses_en)
# DATASET_NAMES=(dev-en)
NOUN_PHRASES_FOLDER=outputs/01-extract-noun-phrases

mkdir -p $OUTPUT_FOLDER


for DATASET_NAME in "${DATASET_NAMES[@]}"
do
    NOUN_PHRASES_PATH=$NOUN_PHRASES_FOLDER/$DATASET_NAME

    python3 extract_unique_lemma_sense.py \
        --input_path $NOUN_PHRASES_PATH/outputs.tsv \
        --output_path $OUTPUT_FOLDER/outputs.tsv
done