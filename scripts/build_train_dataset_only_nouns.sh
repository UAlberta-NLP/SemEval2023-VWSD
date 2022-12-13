DEBUG=$1
OUTPUT_FOLDER=outputs/raganato/only_nouns/train
DATASET_PATH=data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt

if [[ -n "$DEBUG" ]] && [[ "$DEBUG" == "debug" ]]; then
    PYTHONPATH=$(pwd)/consec python3 -m debugpy --wait-for-client --listen 5678 \
        dataset.py build_raganato_dataset \
        --dataset_path=$DATASET_PATH \
        --output_folder=$OUTPUT_FOLDER
else
    PYTHONPATH=$(pwd)/consec python3 dataset.py build_raganato_dataset \
        --dataset_path=$DATASET_PATH \
        --output_folder=$OUTPUT_FOLDER \
        --use_only_nouns
fi
