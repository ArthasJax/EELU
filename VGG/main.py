import os

exp_ReLU = "python3 train.py -a vgg16 --dist-url 'tcp://127.0.0.1:2022' --batch-size 128 --dist-backend 'nccl' --multiprocessing-distributed --learning-rate 0.001 --act_name 'relu' --alpha 0 --beta 0 --eps 0.7 --mode 'channel-wise' --world-size 1 --rank 0 imagenet_2012"
exp_ELU = "python3 train.py -a vgg16 --dist-url 'tcp://127.0.0.1:2022' --batch-size 128 --dist-backend 'nccl' --multiprocessing-distributed --learning-rate 0.0001 --act_name 'ELU' --alpha 1 --beta 1 --eps 0.7 --mode 'channel-wise' --world-size 1 --rank 0 imagenet_2012"
exp2_EELU = "python3 train.py -a vgg16 --dist-url 'tcp://127.0.0.1:2022' --print-freq 1000 --batch-size 128 --dist-backend 'nccl' --multiprocessing-distributed --learning-rate 0.001 --act_name 'EELU' --alpha 0.25 --beta 1 --eps 0.1 --mode 'channel-wise' --world-size 1 --rank 0 imagenet_2012"
exp2_EReLU = "python3 train.py -a vgg16 --dist-url 'tcp://127.0.0.1:2022' --batch-size 128 --dist-backend 'nccl' --multiprocessing-distributed --learning-rate 0.001 --act_name 'EReLU' --alpha 0 --beta 0 --eps 0.4 --mode 'channel-wise' --world-size 1 --rank 0 imagenet_2012"
exp2_EPReLU = "python3 train.py -a vgg16 --dist-url 'tcp://127.0.0.1:2022' --batch-size 128 --dist-backend 'nccl' --multiprocessing-distributed --learning-rate 0.001 --act_name 'EPReLU' --alpha 0.1 --beta 1 --eps 0.4 --mode 'channel-wise' --world-size 1 --rank 0 imagenet_2012"
exp2_MPELU = "python3 train.py -a vgg16 --dist-url 'tcp://127.0.0.1:2022' --batch-size 128 --dist-backend 'nccl' --multiprocessing-distributed --learning-rate 0.001 --act_name 'MPELU' --alpha 1 --beta 1 --eps 0 --mode 'channel-wise' --world-size 1 --rank 0 imagenet_2012"

os.system(exp_ReLU)


