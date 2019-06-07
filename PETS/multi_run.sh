for i in `seq 0 4`
do 
for j in `seq 3 3`
do
python ./scripts/mbexp.py -run $i -window_size $j
done
done

