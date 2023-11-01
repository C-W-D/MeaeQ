#!/bin/bash
# al baselines for main exper
cd ../steal/model_steal
run_seed_arr=(30 31 32 33 34 35 36 37 38 39)
query_num_arr=(191 382 574)
task_name="HATESPEECH" 
# query_num_arr=(201 335 536)
# task_name="SST-2" 
# query_num_arr=(120 200 320)
# task_name="IMDB" 
# query_num_arr=(360 600 960)
# task_name="AGNEWS"
method="AL-US" #'AL-RS;' +'AL-US;'
visible_device=2
vic_model="bert_base_uncased"
steal_model="bert_base_uncased"
vic_model_ck="${vic_model}-victim-model.pkl"
steal_model_ck="${steal_model}-steal-model.pkl"
#___________________________
tokenize_max_length=128
batch_size=32
optimizer="adam"
learning_rate=3e-5
weight_decay=1e-4
num_epochs=10
num_labels=-1
weighted_cross_entropy=False
pool_data_type="None" #'whole;''random_subset;' 'reduced_subset_by_prompt;''reduced_subset_by_prompt_integrate;'
prompt="None"
pool_subsize=-1
epsilon=-1
initial_sample_method="None"
initial_drk_model="None"
al_sample_batch_num=-1
al_sample_method="None" #'random;' +'uncertainty'

if [ ${task_name} = "SST-2" ];then
    tokenize_max_length=128
    batch_size=32
    optimizer="adam"
    learning_rate=3e-5
    weight_decay=1e-4
    num_epochs=10
    num_labels=2
elif [ ${task_name} = "IMDB" ];then
    tokenize_max_length=128
    batch_size=32
    optimizer="adam"
    learning_rate=3e-5
    weight_decay=1e-4
    num_epochs=10
    num_labels=2
    weighted_cross_entropy=True
elif [ ${task_name} = "AGNEWS" ];then
    tokenize_max_length=256
    batch_size=16
    optimizer="adam"
    learning_rate=5e-5
    weight_decay=1e-4
    num_epochs=10
    num_labels=4
elif [ ${task_name} = "HATESPEECH" ];then
    tokenize_max_length=128
    batch_size=32
    optimizer="adam"
    learning_rate=3e-5
    weight_decay=1e-4
    num_epochs=10
    num_labels=2
else
    pass
fi

if [ ${method} = "RS" ];then
    pool_data_type="whole"
    prompt="None"
    pool_subsize=-1
    epsilon=-1
    initial_sample_method="RS"
    initial_drk_model="None"
elif [ ${method} =    "TRF" ];then
    pool_data_type="reduced_subset_by_prompt"
    if [ ${task_name} = "SST-2" ] || [ ${task_name} = "IMDB" ];then
        prompt="This is a movie review."
    elif [ ${task_name} = "AGNEWS" ];then
        prompt="This is a news."
    else
        pass
    fi
    pool_subsize=-1
    epsilon=0.95
    initial_sample_method="random_sentence"
    initial_drk_model="None"
elif [ ${method} =    "DRC" ];then
    pool_data_type="random_subset"
    prompt="None"
    if [ ${task_name} = "SST-2" ] || [ ${task_name} = "IMDB" ];then
        pool_subsize=1766
    elif [ ${task_name} = "AGNEWS" ];then
        pool_subsize=212630 #21264
    else
        pass
    fi
    epsilon=-1
    initial_sample_method="data_reduction_kmeans"
    initial_drk_model="bart-large-mnli"
elif [ ${method} =    "MeaeQ" ];then
    pool_data_type="reduced_subset_by_prompt"
    if [ ${task_name} = "SST-2" ] || [ ${task_name} = "IMDB" ];then
        prompt="This is a movie review."
    elif [ ${task_name} = "AGNEWS" ];then
        prompt="This is a news."
    else
        pass
    fi
    pool_subsize=-1
    epsilon=0.95
    initial_sample_method="data_reduction_kmeans"
    initial_drk_model="bart-large-mnli"
elif [ ${method} =    "AL-RS" ];then
    pool_data_type="random_subset"
    prompt="None"
    pool_subsize=30000
    epsilon=-1
    initial_sample_method="random_sentence"
    initial_drk_model="None"
    al_sample_batch_num=20
    al_sample_method="random"
elif [ ${method} =    "AL-US" ];then
    pool_data_type="random_subset"
    prompt="None"
    pool_subsize=30000
    epsilon=-1
    initial_sample_method="random_sentence"
    initial_drk_model="None"
    al_sample_batch_num=20
    al_sample_method="uncertainty"
else
    pass
fi

for query_num in ${query_num_arr[*]}
do
        for run_seed in ${run_seed_arr[*]}
        do
        output=`command python al_steal.py \
        --victim_model_version ${vic_model} \
        --victim_model_checkpoint ${vic_model_ck} \
        --steal_model_version ${steal_model} \
        --steal_model_checkpoint ${steal_model_ck} \
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
        --query_num ${query_num} \
        --method ${method} \
        --pool_data_type ${pool_data_type} \
        --prompt "${prompt}" \
        --pool_subsize ${pool_subsize} \
        --epsilon ${epsilon} \
        --initial_sample_method ${initial_sample_method} \
        --initial_drk_model ${initial_drk_model} \
        --al_sample_batch_num ${al_sample_batch_num} \
        --al_sample_method ${al_sample_method}`
        done
done