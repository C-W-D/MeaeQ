
import os
import pickle
import random
import numpy as np
from tqdm import tqdm
from copy import copy
import sys

sys.path.append('../../../')
from MeaeQ.utils.config import *


args = ArgParser().get_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.visible_device)
from MeaeQ.utils.tools import *
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import torch.nn as nn
from MeaeQ.models.victim_models import BFSC, RFSC, XFSC
from sentence_transformers import SentenceTransformer, util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids = [0, 1]
print(args.victim_model_version)
if args.victim_model_version == 'bert_base_uncased':
    victim_model = BFSC(args)
elif args.victim_model_version == 'roberta_base':
    victim_model = RFSC(args)
elif args.victim_model_version == 'xlnet_base':
    victim_model = XFSC(args)
if torch.cuda.is_available():
    print("CUDA")
    if torch.cuda.device_count() > 1:
        # models = nn.DataParallel(models)
        victim_model = torch.nn.DataParallel(victim_model, device_ids=device_ids)
    victim_model.to(device)
checkpoint = torch.load(args.saved_model_path + args.task_name + args.victim_model_checkpoint, map_location=device)
victim_model.load_state_dict(checkpoint)
# victim_model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
victim_model.eval()

rate_label = 0.0
label_result = ''
if args.victim_model_version == 'bert_base_uncased':
    tokenizer = BertTokenizer.from_pretrained(args.victim_bert_vocab_path,
                                                   do_lower_case=args.do_lower_case)
elif args.victim_model_version == 'roberta_base':
    tokenizer = RobertaTokenizer.from_pretrained(args.victim_roberta_vocab_path)
elif args.victim_model_version == 'xlnet_base':
    tokenizer = XLNetTokenizer.from_pretrained(args.victim_xlnet_vocab_path)

def generate_query_and_get_predict_label(args=args):
    thief_data = get_pool_data(args)
    print("original thief_data len:", len(thief_data))

    print("generating query and geting labels with api......")
    query = []
    predict_label = []
    sample_num = args.query_num
    data = []
    vocab, freq, probs = read_wikitext103_vocab(args)
    vocab = vocab[:10000]
    if args.initial_sample_method == 'random_sentence':
        index = random.sample(range(len(thief_data)), sample_num)
        for i in index:
            data.append(thief_data[i].strip())
    elif args.initial_sample_method == 'data_reduction_kmeans':
        data = get_reduced_data(thief_data, sample_num, args)
    elif args.initial_sample_method == 'RANDOM' or args.initial_sample_method == 'WIKI':
        result = gen_query_google_baseline(args)
        if args.task_name == 'SST-2' or \
                args.task_name == 'IMDB' or \
                args.task_name == 'AGNEWS' or \
                args.task_name == 'HATESPEECH':
            data = result['sentence']

    for qi in data:
        encoded_pair = tokenizer(qi,
                                    padding='max_length',
                                    truncation=True,
                                    add_special_tokens=True,
                                    max_length=args.tokenize_max_length,
                                    return_tensors='pt')
        if args.victim_model_version == 'bert_base_uncased':
            token_type_ids = to(encoded_pair['token_type_ids'].squeeze(1))
        else:
            token_type_ids = None
        output = victim_model(input_ids=to(encoded_pair['input_ids'].squeeze(1)),
                              token_type_ids=token_type_ids,
                              attention_mask=to(encoded_pair['attention_mask'].squeeze(1)),
                              train_labels=to(torch.zeros(1, args.num_labels)))
        logits = output.logits
        logits = torch.softmax(logits, 1)
        _, test_argmax = torch.max(logits, 1)
        label = test_argmax.squeeze().cpu().data.numpy()
        query.append(qi)
        predict_label.append(label)
    print("sample finish.")
    # if args.do_data_class_balance:
    #     query, predict_label = do_banlace(query, predict_label, args.num_labels)
    #     print("oversample after")

    label_num_cnt = []
    for iii in range(args.num_labels):
        label_num_cnt.append(0)
    for i in predict_label:
        label_num_cnt[i] = label_num_cnt[i] + 1
    global label_result
    for i, j in enumerate(label_num_cnt):
        label_result = label_result + str(j)
        print("%d label count: %d" % (i, j))

    return query, predict_label


def only_get_predict_label(args=args):
    query = []
    predict_label = []
    query_with_label = read_query(args)
    data = query_with_label['sentence']
    for qi in data:
        encoded_pair = tokenizer(qi,
                                    padding='max_length',
                                    truncation=True,
                                    add_special_tokens=True,
                                    max_length=args.tokenize_max_length,
                                    return_tensors='pt')
        if args.victim_model_version == 'bert_base_uncased':
            token_type_ids = to(encoded_pair['token_type_ids'].squeeze(1))
        else:
            token_type_ids = None
        output = victim_model(input_ids=to(encoded_pair['input_ids'].squeeze(1)),
                              token_type_ids=token_type_ids,
                              attention_mask=to(encoded_pair['attention_mask'].squeeze(1)),
                              train_labels=to(torch.zeros(1, args.num_labels)))
        logits = output.logits
        logits = torch.softmax(logits, 1)
        _, test_argmax = torch.max(logits, 1)
        label = test_argmax.squeeze().cpu().data.numpy()
        query.append(qi)
        predict_label.append(label)
    print("sample finish.")
    # if args.do_data_class_balance:
    #     query, predict_label = do_banlace(query, predict_label, args.num_labels)
    #     print("oversample after")

    label_num_cnt = []
    for iii in range(args.num_labels):
        label_num_cnt.append(0)
    for i in predict_label:
        label_num_cnt[i] = label_num_cnt[i] + 1
    global label_result
    for i, j in enumerate(label_num_cnt):
        label_result = label_result + str(j)
        print("%d label count: %d" % (i, j))

    return query, predict_label

if __name__ == "__main__":
    setup_seed(args.run_seed)
    query, predict_label = generate_query_and_get_predict_label()
    # query, predict_label = only_get_predict_label() # if the query have been sampled but the labels are not predicted
    write_query(query, predict_label, args)
