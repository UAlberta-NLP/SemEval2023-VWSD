# Pass in the two files that you wish to evaluate
# Example
# sh score.sh data/text-wsd/mike.txt data/text-wsd/consec.txt 
export cur=`pwd`
cd esc/data/WSD_Evaluation_Framework/Evaluation_Datasets
java Scorer "$cur/$1" "$cur/$2"