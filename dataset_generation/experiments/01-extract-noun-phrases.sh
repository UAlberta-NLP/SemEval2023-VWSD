mkdir -p outputs
OUTPUT_ROOT_FOLDER=$(realpath outputs/$(basename $0 .sh))
DEBUG=$1
DATASET_NAMES=(dev-en test-en semcor_en wngt_examples_en wngt_glosses_en)
# DATASET_NAMES=(dev-en)
# DATASET_NAMES=(test-en)

declare -A DATASET_FOLDERS
DATASET_FOLDERS["dev-en"]="../data/xl-wsd/evaluation_datasets/dev-en"
DATASET_FOLDERS["test-en"]="../data/xl-wsd/evaluation_datasets/test-en"
DATASET_FOLDERS["semcor_en"]="../data/xl-wsd/training_datasets/semcor_en"
DATASET_FOLDERS["wngt_examples_en"]="../data/xl-wsd/training_datasets/wngt_examples_en"
DATASET_FOLDERS["wngt_glosses_en"]="../data/xl-wsd/training_datasets/wngt_glosses_en"

if [[ -n "$DEBUG" ]] && [[ "$DEBUG" == "debug" ]]; then
    echo "DEBUG"        
    DATASET_NAME=test-en
    OUTPUT_FOLDER=$OUTPUT_ROOT_FOLDER/$DATASET_NAME
    mkdir -p $OUTPUT_FOLDER

    python3 -m debugpy --wait-for-client --listen 5678 \
        extract_noun_phrases.py \
        --xml_path "${DATASET_FOLDERS[$DATASET_NAME]}"/$DATASET_NAME.data.xml \
        --sense_keys_path "${DATASET_FOLDERS[$DATASET_NAME]}"/$DATASET_NAME.gold.key.txt \
        --output_path $OUTPUT_FOLDER/outputs.tsv
else
    trap 'jobs -p | xargs -r kill' EXIT

    for DATASET_NAME in "${DATASET_NAMES[@]}"
    do
        OUTPUT_FOLDER=$OUTPUT_ROOT_FOLDER/$DATASET_NAME
        mkdir -p $OUTPUT_FOLDER

        python3 extract_noun_phrases.py \
            --xml_path "${DATASET_FOLDERS[$DATASET_NAME]}"/$DATASET_NAME.data.xml \
            --sense_keys_path "${DATASET_FOLDERS[$DATASET_NAME]}"/$DATASET_NAME.gold.key.txt \
            --output_path $OUTPUT_FOLDER/outputs.tsv &
    done

    wait
fi
