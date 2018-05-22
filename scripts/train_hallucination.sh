#!/bin/bash

cd ../src

gpu=1

sess_per_batch=3
event_per_batch=1000
triplet_per_batch=400
num_seg=3
batch_size=512
n_h=8
n_w=8
n_C=20
n_input=1536

label_num=93
max_epochs=500
static_epochs=250
lr=1e-2
keep_prob=0.5
lambda_l2=0.0
lambda_multimodal=0.5

alpha=0.2
feat="resnet,sensors,segment"
emb_dim=128
network="convrtsn"
optimizer="ADAM"

#name=debug_hallucination
name=hallucination_labelnum${label_num}_lambdamul${lambda_multimodal}

sensors_path='/mnt/work/honda_100h/results/PDDM_sensors_labelnum93_full_20180502-085626/PDDM_sensors_labelnum93_full.ckpt-46500'    # PDDM sensors
segment_path='/mnt/work/honda_100h/results/PDDM_segment_93_20180505-225524/PDDM_segment_93.ckpt-46500'    # PDDM segmentation

python modality_hallucination.py --name $name --lambda_multimodal $lambda_multimodal \
    --n_h $n_h --n_w $n_w --n_C $n_C --n_input $n_input \
    --gpu $gpu  --batch_size $batch_size --feat $feat \
    --triplet_per_batch $triplet_per_batch --max_epochs $max_epochs \
    --sess_per_batch $sess_per_batch --lambda_l2 $lambda_l2 \
    --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim --alpha $alpha \
    --network $network --num_seg $num_seg --keep_prob $keep_prob \
    --sensors_path $sensors_path --optimizer $optimizer --label_num $label_num \
    --segment_path $segment_path

