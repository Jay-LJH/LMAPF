TO_VIS="$1"

for DIR in ${TO_VIS}/*;
do
    python visualization.py \
        --map-file ${DIR}/map.json \
        --path-file ${DIR}/paths.txt
done