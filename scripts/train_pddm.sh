#!/bin/bash

cd ../src

gpu=1

sess_per_batch=3
event_per_batch=1000
triplet_per_batch=400
emb_dim=32
n_h=18
n_w=32
n_C=8
n_input=8
feat="sensors"
#feat="segment"
#n_input=357
network="rtsn"
num_seg=3
batch_size=512
num_negative=3

label_num=93
max_epochs=500
static_epochs=250
lr=1e-2
keep_prob=1.0
lambda_l2=0.
alpha=0.2


name=PDDM_${feat}_labelnum${label_num}

python pddm_model.py --name $name \
    --n_h $n_h --n_w $n_w --n_C $n_C --n_input $n_input \
    --gpu $gpu --batch_size $batch_size --max_epochs $max_epochs \
    --sess_per_batch $sess_per_batch --event_per_batch $event_per_batch \
    --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
    --network $network --num_seg $num_seg --keep_prob $keep_prob --lambda_l2 $lambda_l2 \
    --alpha $alpha --feat $feat --label_num $label_num --triplet_per_batch $triplet_per_batch

