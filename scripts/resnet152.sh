# Low-shot benchmark without generation
for i in 1 2 3 4 5
do
  for j in 1 2 5 10 20
  do
    python ./low_shot.py --lowshotmeta label_idx.json \
      --experimentpath experiment_cfgs/splitfile_{:d}.json \
      --experimentid  $i --lowshotn $j \
      --trainfile features/ResNet152/train.hdf5 \
      --testfile features/ResNet152/val.hdf5 \
      --outdir results \
       --numclasses 244 \
       --maxiter 50000 \
       --batchsize 64 \
      --lr 0.1 --wd 0.01 
  done
done
