OUTPUT_FOLDER=$(realpath outputs/$(basename $0 .sh))
LIBPATH=lib/*:config:. 
INPUT_PATH=outputs/02-extract-unique-lemma-sense/outputs.tsv
OUTPUT_PATH=$OUTPUT_FOLDER/data.tsv

mkdir -p $OUTPUT_FOLDER

javac -cp $LIBPATH RetrieveImages.java && \
java -cp $LIBPATH RetrieveImages \
    $INPUT_PATH \
    $OUTPUT_PATH