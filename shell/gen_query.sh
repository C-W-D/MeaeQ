#!/bin/bash
# generate queries
cd ../steal/model_steal
# query_num_arr=(40 80 120 160 200 240 280 320 360 401 803)
query_num_arr=(15 18)
task_name="HATESPEECH" #'SST-2;' 'IMDB;' 'AGNEWS;' 'HATESPEECH;'
method="MeaeQ" #'RS;' + 'TRF;' +'DRC;' +'MeaeQ;' + 'random-sst2' + 'random-imdb' 'sub-RS'
visible_device=2
victim_model_version="bert_base_uncased" #("bert_base_uncased" "roberta_base" "xlnet_base")
#___________________________
tokenize_max_length=128
batch_size=32
optimizer="adam"
learning_rate=3e-5
weight_decay=1e-4
num_epochs=10
num_labels=-1
pool_data_source="wiki" # imdb sst2
pool_data_type="None" #'whole;''random_subset;' 'reduced_subset_by_prompt;''reduced_subset_by_prompt_integrate;'
prompt="None"
pool_subsize=-1
epsilon=-1
initial_sample_method="None"
initial_drk_model="None"

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
elif [ ${method} = "sub-RS" ];then
    pool_data_type="random_subset"
    prompt="None"
    if [ ${task_name} = "SST-2" ] || [ ${task_name} = "IMDB" ];then
        pool_subsize=1766
    elif [ ${task_name} = "AGNEWS" ];then
        pool_subsize=212630 #21264
    elif [ ${task_name} = "HATESPEECH" ];then
        pool_subsize=1561 #21264
    else
        pass
    fi
    epsilon=-1
    initial_sample_method="RS"
    initial_drk_model="None"
elif [ ${method} = "random-sst2" ];then
    pool_data_source="sst2"
    pool_data_type="whole"
    prompt="None"
    pool_subsize=-1
    epsilon=-1
    initial_sample_method="random_sentence"
    initial_drk_model="None"
elif [ ${method} = "random-imdb" ];then
    pool_data_source="imdb"
    pool_data_type="whole"
    prompt="None"
    pool_subsize=-1
    epsilon=-1
    initial_sample_method="random_sentence"
    initial_drk_model="None"
elif [ ${method} =    "TRF" ];then
    pool_data_type="reduced_subset_by_prompt"
    if [ ${task_name} = "SST-2" ] || [ ${task_name} = "IMDB" ];then
        prompt="This is a movie review."
    elif [ ${task_name} = "AGNEWS" ];then
        prompt="This is a news."
    elif [ ${task_name} = "HATESPEECH" ];then
        prompt="This is a hate speech."
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
    elif [ ${task_name} = "HATESPEECH" ];then
        pool_subsize=1561 #21264
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
    elif [ ${task_name} = "HATESPEECH" ];then
        prompt="This is a hate speech."
    else
        pass
    fi
    pool_subsize=-1
    epsilon=0.95
    initial_sample_method="data_reduction_kmeans"
    initial_drk_model="bart-large-mnli"
else
    pass
fi

for query_num in ${query_num_arr[*]}
do
    output=`command python gen_query.py \
    --victim_model_version ${victim_model_version} \
    --task_name ${task_name} \
    --visible_device ${visible_device} \
    --tokenize_max_length ${tokenize_max_length} \
    --num_labels ${num_labels} \
    --query_num ${query_num} \
    --method ${method} \
    --pool_data_type ${pool_data_type} \
    --prompt "${prompt}" \
    --pool_subsize ${pool_subsize} \
    --epsilon ${epsilon} \
    --initial_sample_method ${initial_sample_method} \
    --initial_drk_model ${initial_drk_model} \
    --pool_data_source ${pool_data_source}`
done