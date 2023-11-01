#!/bin/bash
visible_device=2
victim_model_version="bert_base_uncased" # ("bert_base_uncased" "roberta_base" "xlnet_base")
run_seed=43
weighted_cross_entropy=False
task_name='AGNEWS'
if [ ${task_name} = "SST-2" ];then
    tokenize_max_length=128
    batch_size=32
    optimizer="adam"
    learning_rate=3e-5
    weight_decay=1e-4
    num_epochs=3
    num_labels=2
elif [ ${task_name} = "IMDB" ];then
    tokenize_max_length=128
    batch_size=32
    optimizer="adam"
    learning_rate=3e-5
    weight_decay=1e-4
    num_epochs=3
    num_labels=2
elif [ ${task_name} = "AGNEWS" ];then
    tokenize_max_length=256
    batch_size=16
    optimizer="adam"
    learning_rate=5e-5
    weight_decay=1e-4
    num_epochs=3
    num_labels=4
elif [ ${task_name} = "HATESPEECH" ];then
    tokenize_max_length=128
    batch_size=32
    optimizer="adam"
    learning_rate=3e-5
    weight_decay=1e-4
    num_epochs=3
    num_labels=2
else
    pass
fi
python train.py \
    --victim_model_version ${victim_model_version} \
    --task_name ${task_name} \
    --visible_device ${visible_device} \
    --weighted_cross_entropy ${weighted_cross_entropy} \
    --tokenize_max_length ${tokenize_max_length} \
    --batch_size ${batch_size} \
    --optimizer ${optimizer} \
    --learning_rate ${learning_rate} \
    --weight_decay ${weight_decay} \
    --num_epochs ${num_epochs} \
    --num_labels ${num_labels} \
    --run_seed ${run_seed} \