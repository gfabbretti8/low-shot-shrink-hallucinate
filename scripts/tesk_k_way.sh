for i in 5 10 15 20 25 30 35 40 45 50 55 60 65 70
do
  python ./low_shot.py --lowshotmeta label_idx.json \
    --experimentpath experiment_cfgs/splitfile_3.json \
    --experimentid  $i --lowshotn $j \
    --trainfile features/ResNet152/train.hdf5 \
    --testfile features/ResNet152/val.hdf5 \
    --outdir results \
    --n_way $i \
    --numclasses 244 \
    --maxiter 35000 \
    --batchsize 64 \
   --lr 0.01 --wd 0.001 
done
