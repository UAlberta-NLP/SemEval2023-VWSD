DATASET_NAME=trial
RAGANATO_PATH=outputs/raganato/$DATASET_NAME
OUTPUT_PATH=outputs/consec

echo "Creating a symbolic link from $RAGANATO_PATH to consec/data/tmp"
ln -sf $(realpath $RAGANATO_PATH) consec/data/tmp

mkdir -p consec/outputs/tmp
(
    cd consec && PYTHONPATH=$(pwd) python src/scripts/model/raganato_evaluate.py \
    model.model_checkpoint=checkpoints/consec_wngt_best.ckpt \
    test_raganato_path=data/tmp/$DATASET_NAME \
    model.device=0 \
    hydra.run.dir=outputs/tmp > outputs/tmp/logs
)
mkdir -p $OUTPUT_PATH

echo "Moving consec/outputs/tmp to $OUTPUT_PATH/$DATASET_NAME"
mv consec/outputs/tmp $OUTPUT_PATH/$DATASET_NAME

echo "Removing consec/data/tmp"
rm consec/data/tmp