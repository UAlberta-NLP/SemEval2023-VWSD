DEBUG=$1

if [[ -n "$DEBUG" ]] && [[ "$DEBUG" == "debug" ]]; then
    PYTHONPATH=$(pwd)/consec python3 -m debugpy --wait-for-client --listen 5678 \
        dataset.py build_raganato_dataset \
        --dataset_path=data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt \
        --output_folder=outputs/raganato/train
else
    PYTHONPATH=$(pwd)/consec python3 dataset.py build_raganato_dataset \
        --dataset_path=data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt \
        --output_folder=outputs/raganato/train
fi
