#!/bin/bash
# for cross model extraction, on Hate Speech at query budge * 0.5
cd ./steal/model_steal
query_num_arr=(957 )
run_seed_arr=(30 31 32 33 34 35 36 37 38 39)
task_name="HATESPEECH" 
method="MeaeQ"
visible_device=0
victim_model_version_arr=("bert_base_uncased" "roberta_base" "xlnet_base")
steal_model_version_arr=("bert_base_uncased" "roberta_base" "xlnet_base")
#___________________________
tokenize_max_length=128
batch_size=32
optimizer="adam"
learning_rate=3e-5
weight_decay=1e-4
num_epochs=10
num_labels=-1
pool_data_type="None"
prompt="None"
pool_data_source="RS"
weighted_cross_entropy=False
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

for vic_model in ${victim_model_version_arr[*]}
do
    for steal_model in ${steal_model_version_arr[*]}
    do
        if [ ${steal_model} = "bert_base_uncased" ] && [ ${vic_model} = "bert_base_uncased" ];then
		    continue
		fi
        if [ ${task_name} = "HATESPEECH" ];then
            if [ ${steal_model} != "bert_base_uncased" ];then
                tokenize_max_length=128
                batch_size=16
                optimizer="adam"
                learning_rate=2e-5
                weight_decay=1e-4
                num_epochs=10
                num_labels=2
            fi
        fi
        vic_model_ck="${vic_model}-victim-model.pkl"
        steal_model_ck="${steal_model}-steal-model.pkl"
        echo ${vic_model_ck}
        echo ${steal_model_ck}
        for query_num in ${query_num_arr[*]}
        do
            for run_seed in ${run_seed_arr[*]}
            do
                output=`command python original_steal.py \
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
                --al_sample_method ${al_sample_method} \
                --pool_data_source ${pool_data_source}`
            done
        done
    done
done