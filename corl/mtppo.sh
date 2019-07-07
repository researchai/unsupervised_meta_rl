echo "$1"

for i in $(seq $2 $3)
do
    PYTHONPATH=/root/code/garage:/root/code/garage/metaworld:/root/code/garage/corl/baby python ./mtppo_easy_mode_test.py --snapshot_dir $1 --itr $i
done
