#!/bin/bash -l

# Loop over files and calculate semantic similarity
for file in data/*processed_toy_example.tsv; do
# Run the Python script for each file
    fname=`basename $file`
    model=`echo $fname | sed "s/_processed.*//"`
    model_name_col="${model}_default Name"
    GO_name_col='Term_Description'
    echo "Processing $file"
    echo "namecol1 $GO_name_col"
    echo "namecol2 $model_name_col"
    python calculate_sem_sim.py --inputFile $file --nameCol1 "$GO_name_col" --nameCol2 "$model_name_col"

done
