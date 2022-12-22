OUTPUT_FOLDER=$(realpath outputs/$(basename $0 .sh))
LIBPATH=lib/*:config:. 
INPUT_PATH=outputs/03-retrieve-synset-ids/unique.txt
OUTPUT_PATH=$OUTPUT_FOLDER/data.tsv

mkdir -p $OUTPUT_FOLDER

javac -cp $LIBPATH RetrieveURLs.java && \
java -cp $LIBPATH RetrieveURLs \
    $INPUT_PATH \
    $OUTPUT_PATH