# Training ViPT
# python tracking/train.py --script EventTPT --config deep_rgbd --save_dir ./output --mode multiple --nproc_per_node 2
# python tracking/train.py --script EventTPT --config deep_rgbt --save_dir ./output --mode multiple --nproc_per_node 2
python tracking/train.py --script vipt --config deep_rgbe --save_dir ./output --mode single #--nproc_per_node 1


# Training ViPT-shaw
#python tracking/train.py --script EventTPT --config shaw_rgbd --save_dir ./output --mode multiple --nproc_per_node 2
#python tracking/train.py --script EventTPT --config shaw_rgbt --save_dir ./output --mode multiple --nproc_per_node 2
#python tracking/train.py --script EventTPT --config shaw_rgbe --save_dir ./output --mode multiple --nproc_per_node 2
