echo "Performing COMET eval for $1..." &&
comet-score -s "$1/src.txt" -t "$1/tgt.txt" --model Unbabel/wmt22-cometkiwi-da