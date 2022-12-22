OUTPUT_FOLDER=$(realpath outputs/$(basename $0 .sh))
LIBPATH=lib/*:config:. 
INPUT_PATH=outputs/02-extract-unique-lemmas/outputs.tsv
OUTPUT_PATH=$OUTPUT_FOLDER/data.tsv

mkdir -p $OUTPUT_FOLDER

javac -cp $LIBPATH ExtractSenses.java && \
java -cp $LIBPATH ExtractSenses \
    $INPUT_PATH \
    $OUTPUT_FOLDER/data.tsv \
    $OUTPUT_FOLDER/unique.txt