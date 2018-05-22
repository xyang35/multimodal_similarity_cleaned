#!/bin/bash

cd ../src


##############################################################################


feat="resnet"
#feat="sensors"
#feat="segment_down"
#feat="segment"

num_seg=3
variable_name="modality_core/"
label_type="goal"
#label_type="stimuli"
gpu=0


if [ $feat == "resnet" ]
then
    network="convrtsn"
    emb_dim=128
    n_h=8
    n_w=8
    n_C=20
    n_input=1536
elif [ $feat == "sensors" ]
then
    network="rtsn"
    emb_dim=32
    n_input=8
    # void input
    n_h=8
    n_w=8
    n_C=20
elif [ $feat == "segment_down" ]
then
    network="convrtsn"
    emb_dim=128
    n_h=18
    n_w=32
    n_C=8
    n_input=17
elif [ $feat == "segment" ]
then
    network="rtsn"
    emb_dim=32
    n_input=357
    # void input
    n_h=18
    n_w=32
    n_C=8
fi

#model_path='/mnt/work/honda_100h/final_results/base200_labelnum9_20180513-012341/base200_labelnum9.ckpt-6002'
#model_path='/mnt/work/honda_100h/final_results/base400_labelnum9_20180513-223526/base400_labelnum9.ckpt-5999'
#model_path='/mnt/work/honda_100h/results/multimodal_lambdamul0.1_labelnum9_20180514-230424/multimodal_lambdamul0.1_labelnum9.ckpt-5999'
#model_path='/mnt/work/honda_100h/results/multimodal_full_lambdamul0.1_labelnum9_0.3_20180515-015310/multimodal_full_lambdamul0.1_labelnum9_0.3.ckpt-4876'
model_path='/mnt/work/honda_100h/results/multimodal_full_lambdamul0.1_labelnum9_20180515-004621/multimodal_full_lambdamul0.1_labelnum9.ckpt-5478'

#python evaluate_hallucination.py --model_path $model_path --feat $feat \
python evaluate_model.py --model_path $model_path --feat $feat \
                   --network $network --num_seg $num_seg \
                   --label_type $label_type \
                   --n_h $n_h --n_w $n_w --n_C $n_C --n_input $n_input \
                   --gpu $gpu --emb_dim $emb_dim  --variable_name $variable_name #--no_transfer

