#!/bin/bash

cd ../src

gpu=1

num_threads=2
sess_per_batch=3
num_seg=3
batch_size=512
metric="squaredeuclidean"

label_num=9
max_epochs=2000
static_epochs=1000
multimodal_epochs=0
lr=1e-2
keep_prob=0.5
lambda_l2=0.0
lambda_multimodal=2000
alpha=0.2

triplet_per_batch=200
triplet_select="facenet"
feat="resnet,sensors,segment"
emb_dim=128
network="convrtsn"
task="supervised"

name=Crosspred_both_labelnum${label_num}_lambdamul${lambda_multimodal}

#sensors_path='/mnt/work/honda_100h/results/PDDM_sensors_labelnum93_full_20180502-085626/PDDM_sensors_labelnum93_full.ckpt-46500'    # PDDM sensors
#segment_path='/mnt/work/honda_100h/results/PDDM_segment_93_20180505-225524/PDDM_segment_93.ckpt-46500'    # PDDM segmentation
segment_path='/mnt/work/honda_100h/results/PDDM_segment_labelnum9_20180509-164911/PDDM_segment_labelnum9.ckpt-4500'    # PDDM segment, label_num=9
sensors_path='/mnt/work/honda_100h/results/PDDM_sensors_labelnum9_20180509-164952/PDDM_sensors_labelnum9.ckpt-4500'    # PDDM sensors, label_num=9

#model_path='/mnt/work/honda_100h/results/Crosspred_both_labelnum9_lambdamul100_20180510-024532/Crosspred_both_labelnum9_lambdamul100.ckpt-5664'
#model_path='/mnt/work/honda_100h/results/Crosspred_both_labelnum9_lambdamul500_20180510-024454/Crosspred_both_labelnum9_lambdamul500.ckpt-6648'
#model_path='/mnt/work/honda_100h/results/Crosspred_both_labelnum9_lambdamul10_20180510-110712/Crosspred_both_labelnum9_lambdamul10.ckpt-6039'
#model_path='/mnt/work/honda_100h/results/Crosspred_both_labelnum9_lambdamul1000_20180510-110734/Crosspred_both_labelnum9_lambdamul1000.ckpt-6537'

python multitask_cross_prediction.py --name $name --lambda_multimodal $lambda_multimodal \
    --gpu $gpu --num_threads $num_threads --batch_size $batch_size --feat $feat \
    --triplet_per_batch $triplet_per_batch --max_epochs $max_epochs --alpha $alpha \
    --triplet_select $triplet_select --sess_per_batch $sess_per_batch \
    --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
    --metric $metric --network $network --num_seg $num_seg --keep_prob $keep_prob \
    --label_num $label_num --multimodal_epochs $multimodal_epochs --sensors_path $sensors_path \
    --segment_path $segment_path --task $task #--model_path $model_path
