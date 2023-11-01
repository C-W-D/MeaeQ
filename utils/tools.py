import pickle
import sys

from pyclustering.utils import distance_metric, type_metric

sys.path.append('../..')
from MeaeQ.utils.config import *

args = ArgParser().get_parser()
import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.visible_device)
import time

from MeaeQ.models.extracted_models import *
from torch.autograd import Variable
import numpy as np
np.bool = np.bool_
import random
import torch
from sklearn.cluster import KMeans
import faiss
from sentence_transformers import SentenceTransformer, util
from MeaeQ.utils.MyDataset import *
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids = [0, 1]


def torch_cuda_cache_clean():
    cnt = 30
    while cnt > 0:
        torch.cuda.empty_cache()
        cnt = cnt - 1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def to(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def sentence_length_calculate(sentence_list, alpha=0.95):
    # Calculate the maximum length of word segmentation
    text_len_list = []

    for i in sentence_list:
        text_len_list.append(len(i))
    total = len(text_len_list)
    newl = sorted(text_len_list)
    maxlen = newl[int(total * alpha)]
    return maxlen


def update_file(file):
    file = file.replace()
    return file


def build_pool_file_content(args):
    root = args.root
    seed = args.seed
    task_name = args.task_name
    pool_data_source = args.pool_data_source
    pool_data_type = args.pool_data_type
    prompt = args.prompt.replace(' ', '-')
    pool_subsize = args.pool_subsize

    seed_fn = 'seed=' + str(seed)
    task_name_fn = 'task_name=' + task_name
    pool_data_source_fn = 'pool_data_source=' + pool_data_source
    pool_data_type_fn = 'pool_data_type=' + pool_data_type
    prompt_fn = 'prompt=' + prompt
    pool_subsize_fn = 'pool_subsize=' + str(pool_subsize)

    root_file = root + seed_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + task_name_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + pool_data_source_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + pool_data_type_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + prompt_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + pool_subsize_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    pool_data_file = root_file + 'pool_data_sentence.pickle'
    pool_data_pb_file = root_file + 'pool_data_pb.pickle'
    return root_file, pool_data_file, pool_data_pb_file


def build_query_file_content(root_file, args):
    epsilon = args.epsilon
    initial_sample_method = args.initial_sample_method
    initial_drk_model = args.initial_drk_model
    query_num = args.query_num

    if epsilon == -1:
        epsilone_fn = 'epsilon=-1'
    else:
        epsilone_fn = 'epsilon=' + str(epsilon)
    initial_sample_method_fn = 'initial_sample_method=' + initial_sample_method
    initial_drk = 'initial_drk_model=' + initial_drk_model.replace(" ", '-')
    query_num_fn = 'query_num=' + str(query_num)

    root_file = root_file + epsilone_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + initial_sample_method_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + initial_drk + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + query_num_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)

    query_content_data_file = root_file + 'query_content.pickle'
    query_label_data_file = root_file + 'query_label.pickle'

    return query_content_data_file, query_label_data_file


def write_query(query, predict_label, args):
    root_file, _, __ = build_pool_file_content(args)
    query_content_data_file, query_label_data_file = build_query_file_content(root_file, args)
    if args.victim_model_version == 'bert_base_uncased':
        pass
    else:
        query_label_data_file = query_label_data_file.replace("query_label", "query_label_" + args.victim_model_version)
    with open(query_content_data_file, 'wb') as f:
        pickle.dump(query, f)
    with open(query_label_data_file, 'wb') as f:
        pickle.dump(predict_label, f)


def read_query(args):
    """
        this func only provide the query (pre-constructed) and the predicted label from vic (bert-base-uncased)
        given the different vic model, we write a new func named read_query_update
    """
    root_file, _, __ = build_pool_file_content(args)
    query_content_data_file, query_label_data_file = build_query_file_content(root_file, args)
    with open(query_content_data_file, 'rb') as f:
        query = pickle.load(f)
    with open(query_label_data_file, 'rb') as f:
        label = pickle.load(f)

    data = {
        'sentence': query,
        'label': label
    }
    return data


def read_query_update(args):
    """
        query_label_data_file can be related with different vic_model
        the only differnce between this func with read_query() is that
        read_query() only provide label with vic (bert-base-uncased)

        if you want to read the pre-constructed query but not the label,
        you can just use the read_query() to get the query and ignore the label

        this func can be used when train the extracted model
    """
    root_file, _, __ = build_pool_file_content(args)
    query_content_data_file, query_label_data_file = build_query_file_content(root_file, args)
    if args.victim_model_version != 'bert_base_uncased':
        query_label_data_file = query_label_data_file.replace("query_label", "query_label_" + args.victim_model_version)
    with open(query_content_data_file, 'rb') as f:
        query = pickle.load(f)
    with open(query_label_data_file, 'rb') as f:
        label = pickle.load(f)


    data = {
        'sentence': query,
        'label': label
    }
    return data


def write_pool_data(pool_data, args):
    _, pool_data_file, __ = build_pool_file_content(args)
    with open(pool_data_file, 'wb') as f:
        pickle.dump(pool_data, f)


def read_pool_data(args):
    _, pool_data_file, __ = build_pool_file_content(args)
    with open(pool_data_file, 'rb') as f:
        pool_data = pickle.load(f)
    return pool_data


def write_pool_data_pb(pool_data_pb, args):
    _, __, pool_data_pb_file = build_pool_file_content(args)
    with open(pool_data_pb_file, 'wb') as f:
        pickle.dump(pool_data_pb, f)


def read_pool_data_pb(args):
    _, __, pool_data_pb_file = build_pool_file_content(args)
    with open(pool_data_pb_file, 'rb') as f:
        pool_data_pb = pickle.load(f)
    return pool_data_pb


def read_wikitext103_vocab(args):
    with open(args.wikitext103_vocab_file, 'rb') as f:
        vocab_freq_pb = pickle.load(f)
    freq = []
    probs = []
    vocab = []
    for x in vocab_freq_pb:
        vocab.append(x[0])
        freq.append(x[1][0])
        probs.append(x[1][1])
    return vocab, freq, probs


def read_wikitext103_sentence_pool(args):
    with open(args.wiki_path, "r") as f:
        wikitext103_sentence = f.read().strip().split("\n")
    return wikitext103_sentence


def random_sample_uniform(args):
    """
        Uniformly and randomly extract words of specified length 
        from the top 10,000 words in the dictionary and form sentences
    """
    vocab, freq, probs = read_wikitext103_vocab(args)
    wikitext_sentence = read_wikitext103_sentence_pool(args)
    vocab = vocab[:10000]
    freq = freq[:10000]
    probs = probs[:10000]
    length_list = [
        len(wikitext_sentence[i])
        for i in random.sample(range(len(wikitext_sentence)), args.query_num)
    ]
    result = [
        ' '.join([random.choice(vocab) for _ in range(i)])
        for i in length_list
    ]
    return result


def random_sample_freq(args):
    """
        Obey the univariate probability distribution to 
        extract words of specified length from the top 10,000 words in the dictionary and form sentences.
    """
    vocab, freq, probs = read_wikitext103_vocab(args)
    wikitext_sentence = read_wikitext103_sentence_pool(args)
    length_list = [
        len(wikitext_sentence[i])
        for i in random.sample(range(len(wikitext_sentence)), args.query_num)
    ]
    result = []
    for seq_length in length_list:
        bow_sent = np.random.multinomial(seq_length, probs)
        nonzero_indices = np.nonzero(bow_sent)[0]
        bow_indices = []
        for index, freq in zip(nonzero_indices, bow_sent[nonzero_indices]):
            bow_indices.extend([index for _ in range(freq)])
        assert len(bow_indices) == seq_length
        random.shuffle(bow_indices)
        result.append(' '.join([vocab[i] for i in bow_indices]))
    return result


def wiki_sample_sentence_uniform(args):
    """
        Choose a sentence at random from wikitext-103.
        Words in the sentence that do not belong to the top 
        10,000 wikitext-103 vocabulary are replaced with uniformly 
        randomly selected words from this vocabulary.
    """
    vocab, freq, probs = read_wikitext103_vocab(args)
    vocab = vocab[:10000]
    wikitext_sentence = read_wikitext103_sentence_pool(args)
    result = []
    for i in range(args.query_num):
        t = random.choice(wikitext_sentence)
        t = ' '.join([
            random.choice(vocab) if j not in vocab
            else j
            for j in t.split(' ')
        ])
        result.append(t)
    return result


def wiki_sample_paragraph_uniform(args):
    """
        Select a random paragraph from wikitext-103.
    """
    wikitext_paragraph = read_wikitext103_sentence_pool(args)
    result = [random.choice(wikitext_paragraph) for _ in range(args.query_num)]
    return result


def gen_query_google_baseline(args):
    # By default, WIKI is used as the pool source and the pool type is whole.
    result = {}
    if args.task_name == 'SST-2' or \
            args.task_name == 'IMDB' or \
            args.task_name == 'AGNEWS' or \
            args.task_name == 'HATESPEECH':
        if args.initial_sample_method.find('RANDOM') != -1:
            query = random_sample_uniform(args)
        elif args.initial_sample_method.find('WIKI') != -1:
            query = wiki_sample_sentence_uniform(args)

        result = {
            'sentence': query
        }

    return result


def do_banlace(query, label, num_labels):
    # repeat sampling for imbalanced labels
    label_cnt = []
    sentence = []
    for i in range(num_labels):
        label_cnt.append([])
        sentence.append([])
    for ii, i in enumerate(label):
        sentence[i].append(query[ii])
        label_cnt[i].append(i)
    max_v = -1
    max_index = -1
    print("oversample before")
    for ii, i in enumerate(label_cnt):
        length = len(label_cnt[ii])
        print("%d label count: %d" % (ii, length))
        if length > max_v:
            max_v = length
            max_index = ii
    print("max index:", max_index)
    print("max v:", max_v)
    # the remainder label will copy itself until the num is up to max_v
    for ii, i in enumerate(label_cnt):
        print(ii, max_index, ii == max_index)
        if ii == max_index:
            continue
        diff = max_v - len(label_cnt[ii])
        div = diff // len(label_cnt[ii])
        mod = diff % len(label_cnt[ii])
        for j in range(div):
            query.extend(sentence[ii])
            label.extend(label_cnt[ii])
        query.extend(sentence[ii][:mod])
        label.extend(label_cnt[ii][:mod])
    return query, label


def get_pool_data(args):
    if args.pool_data_source == 'wiki':
        thief_dataset = args.wiki_path
    elif args.pool_data_source == 'imdb':
        thief_dataset = args.imdb_path
    elif args.pool_data_source == 'sst2':
        thief_dataset = args.sst2_path
    with open(thief_dataset, "r") as f:
        thief_data = f.read().strip().split("\n")
    #     thief_data = thief_data[:200]
    subsize = args.pool_subsize
    vocab, freq, probs = read_wikitext103_vocab(args)
    vocab = vocab[:10000]
    if args.pool_data_type == 'whole':
        pass
    elif args.pool_data_type == 'random_subset':
        tindex = random.sample(range(len(thief_data)), subsize)
        temp_data = []

        for i in tindex:
            temp_data.append(thief_data[i])
        thief_data = temp_data

    elif args.pool_data_type == 'reduced_subset_by_prompt':
        _, pool_data_file, __ = build_pool_file_content(args)
        if not os.path.exists(pool_data_file):
            print("pool data file")
            nli_model = AutoModelForSequenceClassification.from_pretrained(args.bart_large_mnli_path)
            tokenizer = AutoTokenizer.from_pretrained(args.bart_large_mnli_path)
            if torch.cuda.is_available():
                print("CUDA")
                nli_model.to(device)
            #         temp_data = []
            logit = []
            pool_data = []
            print("begin predict by bart")

            start_time = time.time()
            hypothesis = args.prompt
            print("hypothesis: ", args.prompt)

            for i, ix in enumerate(thief_data):

                premise = ix

                # run through model pre-trained on MNLI
                if args.task_name == 'SNLI' and hypothesis.find("between") != -1:
                    token_list = premise.strip().split(' ')
                    for _ in range(3):
                        random_index = random.randint(0, len(token_list) - 1)
                        token_list[random_index] = random.choice(vocab)
                    snli_hy = ' '.join(token_list)
                    premise = premise + '</s><s>' + snli_hy

                pool_data.append(premise)

                x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                                     truncation_strategy='only_first')
                logits = nli_model(x.to(device))[0]
                #                 logits = nli_model(x)[0]
                """
                nli_model(x)：
                Seq2SeqSequenceClassifierOutput(loss=None, 
                logits=tensor([[-2.4954,  0.9106,  0.5426]], 
                grad_fn=<AddmmBackward>), 
                past_key_values
                encoder_hidden_states=None, encoder_attentions=None)
                """
                # we throw away "neutral" (dim 1) and take the probability of
                # "entailment" (2) as the probability of the label being true
                entail_contradiction_logits = logits[:, [0, 2]]
                probs = entail_contradiction_logits.softmax(dim=1)
                prob_label_is_true = probs[0, 1]
                #                 print(prob_label_is_true)
                logit.append(prob_label_is_true.cpu().data.numpy().tolist())
                #                 print(premise)
                #             if prob_label_is_true > args.hypothesis_data_distribution_epsilon:
                #                 temp_data.append(premise)
                #         thief_data = temp_data
                #         print(logit)
                if i % 100 == 0:
                    duration = time.time() - start_time
                    seconds = int(duration)
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    print("%d / %d cost time: %d:%02d:%02d" % (i, len(thief_data), h, m, s))
                    start_time = time.time()

                if i % 100000 == 0:
                    write_pool_data_pb(logit, args)
            print("finish...")
            write_pool_data(pool_data, args)
            write_pool_data_pb(logit, args)
        else:
            print("existed")
            thief_data = read_pool_data(args)
            thief_data_pb = read_pool_data_pb(args)
            temp_data = []
            for i, ix in enumerate(thief_data_pb):
                if ix >= args.epsilon:
                    temp_data.append(thief_data[i])
            thief_data = temp_data
            print("thief data size:", len(thief_data))

    return thief_data


# the following code: data reduction based on clusterin
nli_model = AutoModelForSequenceClassification.from_pretrained(args.bart_large_mnli_path)
tokenizer = AutoTokenizer.from_pretrained(args.bart_large_mnli_path)
if torch.cuda.is_available():
    print("CUDA")
    nli_model.to(device)
sim_model = []
flag = 1
def cosine_distance(a, b):
    # dot product
    # a b have been normalized
    similiarity = np.dot(a, b.T)
    dist = 1. - similiarity
    return dist

def get_reduced_data(thief_data, batch_num, args):
    """
    function:
        sample a batch of the most valuable data from thief dataset
    param:
        @ thief_data: unlabeled thief dataset, a sentences list
        @ batch_num: # of the query seed size
    return:
        @ Smin: the reduced data, a sentence list
    """
    sentence_embed = []
    for i, x in enumerate(thief_data):
        x = x.replace('</s><s>', '')
        if flag == 0:
            sentence_embed.append(sim_model.encode(x))
        elif flag == 1:
            xx = tokenizer.encode(x, None, return_tensors='pt',
                                  truncation_strategy='only_first').to(device)
            xx = nli_model(xx)
            sentence_embed.append(
                xx.decoder_hidden_states[-1][:, -1, :].squeeze(0).detach().cpu().data.numpy().tolist()
            )
    Smin = []
    sentence_embed = np.array(sentence_embed).astype(np.float64)
    sentence_embed = [i / np.linalg.norm(i) for i in sentence_embed]

    n_clusters = batch_num
    initial_centers = kmeans_plusplus_initializer(sentence_embed, n_clusters, random_state=args.seed).initialize()
    metric = distance_metric(type_metric.USER_DEFINED, func=cosine_distance)
    kmeans_instance = kmeans(sentence_embed, initial_centers, metric=metric)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    cluster_centers = kmeans_instance.get_centers()
    cluster_centers = [i / np.linalg.norm(i) for i in cluster_centers]
    select_data_index = []
    for i, ix in enumerate(clusters):
        # print("cluster", ix)
        cluster_center = cluster_centers[i]
        data_index = clusters[i]
        data_embed = []
        rmap = {}
        # print("data index", data_index)
        for j, jx in enumerate(data_index):
            data_embed.append(sentence_embed[jx])
            rmap[j] = jx

        d = len(cluster_center)  # dimension
        nb = len(data_embed)  # database size
        nlist = 1
        xb = np.array(data_embed).astype('float32')
        faiss.normalize_L2(xb)
        index = faiss.IndexFlatIP(d)  # IndexFlatL2 & IndexFlatIP -> euc & dot pro
        index = faiss.IndexIVFFlat(index, d, nlist, faiss.METRIC_INNER_PRODUCT)
        # index = faiss.IndexIDMap(index)
        index.train(xb)
        index.add_with_ids(xb, np.array(data_index))
        cluster_center = np.array(cluster_center).astype('float32')
        D, I = index.search(np.array([cluster_center, ]), 1)
        I = I[0]
        Smin.append(thief_data[I[0]].strip())
        select_data_index.append(I[0])
    return Smin


if __name__ == '__main__':
    query = read_query(args)