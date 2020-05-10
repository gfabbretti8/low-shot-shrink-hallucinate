for i in 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70
do
  for j in 1 5
  do
    python ./low_shot.py --lowshotmeta label_idx.json \
      --experimentpath experiment_cfgs/splitfile_{:d}.json \
      --experimentid  $i --lowshotn $j \
      --trainfile features/ResNet152/train.hdf5 \
      --testfile features/ResNet152/val.hdf5 \
      --outdir results \
      --n_way $i \
      --lr 0.1 --wd 0.01 \
      --testsetup 1
  done
done
