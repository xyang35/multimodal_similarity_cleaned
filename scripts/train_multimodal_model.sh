#!/bin/bash

cd ../src

gpu=1

sess_per_batch=3
event_per_batch=1000
triplet_per_batch=200
num_seg=3
batch_size=512

label_num=93
max_epochs=500
static_epochs=250
multimodal_epochs=0
lr=1e-2
keep_prob=0.5
lambda_l2=0.0
lambda_multimodal=0.1

alpha=0.2
feat="resnet,sensors,segment"
emb_dim=128
network="convrtsn"
optimizer="ADAM"

name=multimodal_full_lambdamul${lambda_multimodal}_labelnum${label_num}
#name=debug

sensors_path='/mnt/work/honda_100h/results/PDDM_sensors_labelnum93_full_20180502-085626/PDDM_sensors_labelnum93_full.ckpt-46500'    # PDDM sensors
segment_path='/mnt/work/honda_100h/results/PDDM_segment_93_20180505-225524/PDDM_segment_93.ckpt-46500'    # PDDM segmentation

python multimodal_model.py --name $name --lambda_multimodal $lambda_multimodal \
    --gpu $gpu --batch_size $batch_size --feat $feat --multimodal_epochs $multimodal_epochs \
    --triplet_per_batch $triplet_per_batch --max_epochs $max_epochs \
    --sess_per_batch $sess_per_batch --lambda_l2 $lambda_l2 --label_num $label_num \
    --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim --alpha $alpha \
    --network $network --num_seg $num_seg --keep_prob $keep_prob \
    --optimizer $optimizer --event_per_batch $event_per_batch \
    --sensors_path $sensors_path --segment_path $segment_path --no_joint

