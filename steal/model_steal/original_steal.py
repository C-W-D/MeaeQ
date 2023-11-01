import math
import sys
sys.path.append('../../..')
import random
import numpy as np
from tqdm import tqdm
from MeaeQ.utils.config import *
args = ArgParser().get_parser()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.visible_device)
print(args.visible_device)

from MeaeQ.utils.tools import *
from MeaeQ.utils.DataProcessor import *
from MeaeQ.utils.MyDataset import *
import torch
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, BertModel
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from MeaeQ.models.victim_models import BFSC, RFSC, XFSC
from MeaeQ.models.extracted_models import BPC, RPC, XPC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids = [0, 1]
vac = 0.0
acc = 0.0
seed = args.run_seed
aggrement = 0.0
weights = []
print(seed)
print(args.query_num)
dir = 'log/' + args.task_name + '-' + args.method + '-vic=' + args.victim_model_version + '-steal=' + args.steal_model_version
if not os.path.exists(dir):
    os.mkdir(dir)
all_result = dir + '/all_result.txt'
output_file = dir + '/query_num=' + str(args.query_num) + '-seed=' + str(args.run_seed) + '-device=' + str(args.visible_device) + '.txt'
sck = args.steal_model_checkpoint
if sck[0] != '-':
    sck = '-' + sck
best_validate_dir = args.saved_model_path + str(args.run_seed) + args.method + str(args.query_num) + str(args.visible_device) + args.task_name + sck
def split_steal_data(steal_data):
    """
        steal_data: dict
    """

    train_test_key = steal_data['sentence']
    train_test_value = steal_data['label']
    tra_key, ts_key, tra_value, ts_value = \
        train_test_split(train_test_key,
                         train_test_value,
                         test_size=0.1)

    train = {
        'sentence': tra_key,
        'label': tra_value
    }
    val = {
        'sentence': ts_key,
        'label': ts_value
    }
    return train, val


def train_steal_model(steal_train, steal_val):
    train_dataset = StealQueryDataset(steal_train, 'with_label', args)
    val_dataset = StealQueryDataset(steal_val, 'with_label', args)
    # test_dataset = StealQueryDataset(steal_test, 'with_label')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    validate_loader = DataLoader(dataset=val_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    # test_loader = DataLoader(dataset=test_dataset,
    #                          batch_size=args.batch_size,
    #                          shuffle=False)
    print('training steal models')
    # config = BertConfig.from_pretrained(steal_bert_path, num_labels=2, hidden_dropout_prob=0.2)
    # models = BertForSequenceClassification.from_pretrained(steal_bert_path, config=config)
    print("steal model: ", args.steal_model_version)
    if args.steal_model_version == 'bert_base_uncased':
        model = BPC(args)
    elif args.steal_model_version == 'roberta_base':
        model = RPC(args)
    elif args.steal_model_version == 'xlnet_base':
        model = XPC(args)

    if torch.cuda.is_available():
        print("CUDA")
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            print("multi gpus")
        model.to(device)

    # optimizer
    global optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, list(model.parameters())),
            lr=args.learning_rate
        )

    # scheduler
    scheduler = None
    # if args.scheduler == 'Step':
    #     scheduler = StepLR(optimizer, step_size=10, gamma=0.8, last_epoch=-1)
    # elif args.scheduler == 'MultiStep':
    #     scheduler = MultiStepLR(optimizer, milestones=[10, 30, 80], gamma=0.8, last_epoch=-1)
    # elif args.scheduler == 'Exponential':
    #     scheduler = ExponentialLR(optimizer, gamma=0.5, last_epoch=-1)
    # elif args.scheduler == 'CosineAnnealing':
    #     scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.1 * args.learning_rate, last_epoch=-1)
    # elif args.scheduler == 'self-definition':
    #     scheduler = None

    # loss
    if args.loss == 'cross-entropy-loss':
        criterion = nn.CrossEntropyLoss()

    print('training steal models on query dataset...')

    if not os.path.exists(args.saved_model_path):
        os.mkdir(args.saved_model_path)
    train_loss = []
    train_acc = []
    vali_loss = []
    vali_acc = []
    best_vacc = -1
    if args.weighted_cross_entropy:
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    Epochs = args.num_epochs
    for epoch in range(Epochs):
        if scheduler is None:
            p = float(epoch) / Epochs
            lr = args.learning_rate / (1. + 10 * p) ** 0.75
            optimizer.lr = lr

        train_cost_vector = []
        train_acc_vector = []
        model.train()

        # loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        # for step, train_data in loop:
        for step, train_data in enumerate(train_loader):
            train_text = train_data[0]
            train_labels = to(train_data[1])
            input_ids = to(train_text['input_ids'].squeeze(1))
            attention_mask = to(train_text['attention_mask'].squeeze(1))
            if args.steal_model_version == 'bert_base_uncased':
                token_type_ids = to(train_text['token_type_ids'].squeeze(1))
            else:
                token_type_ids = None
            output, pooler_output = model(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          attention_mask=attention_mask,
                                          train_labels=train_labels)
            logits = output.logits
            # logits = torch.softmax(logits, 1)
            # class_loss = output.loss
            class_loss = criterion(logits, train_labels)
            optimizer.zero_grad()
            class_loss.backward()
            optimizer.step()
            _, argmax = torch.max(logits, 1)

            accuracy = (train_labels == argmax.squeeze()).float().mean()

            train_cost_vector.append(class_loss.item())
            train_acc_vector.append(accuracy.item())

            # loop.set_description('Epoch [%d/%d]' % (epoch + 1, Epochs))
            # loop.set_postfix(loss=class_loss.item(),
            #                  train_acc=accuracy.item())
        tloss = np.mean(train_cost_vector)
        tacc = np.mean(train_acc_vector)
        train_loss.append(tloss)
        train_acc.append(tacc)
        if scheduler is not None:
            scheduler.step()
        validate_cost_vector = []
        validate_acc_vector = []
        model.eval()
        with torch.no_grad():
            for validate_data in validate_loader:
                validate_text = validate_data[0]
                validate_labels = to(validate_data[1])
                input_ids = to(validate_text['input_ids'].squeeze(1))
                attention_mask = to(validate_text['attention_mask'].squeeze(1))
                if args.steal_model_version == 'bert_base_uncased':
                    token_type_ids = to(validate_text['token_type_ids'].squeeze(1))
                else:
                    token_type_ids = None
                output, pooler_output = model(input_ids=input_ids,
                                              token_type_ids=token_type_ids,
                                              attention_mask=attention_mask)
                logits = output.logits
                logits = torch.softmax(logits, 1)
                _, argmax = torch.max(logits, 1)
                accuracy = (validate_labels == argmax.squeeze()).float().mean()

                validate_acc_vector.append(accuracy.item())

        vloss = np.mean(validate_cost_vector)
        vacc = np.mean(validate_acc_vector)
        vali_loss.append(vloss)
        vali_acc.append(vacc)
        print('Epoch [%d/%d],  train_loss: %.4f, train_acc: %.4f, vali_acc: %.4f'
              % (epoch + 1, Epochs, tloss, tacc, vacc))
        if vacc > best_vacc:
            best_vacc = vacc
            torch.save(model.state_dict(), best_validate_dir)

    #     print('testing steal models on query dataset...')
    if args.steal_model_version == 'bert_base_uncased':
        model = BPC(args)
    elif args.steal_model_version == 'roberta_base':
        model = RPC(args)
    elif args.steal_model_version == 'xlnet_base':
        model = XPC(args)
    if torch.cuda.is_available():
        print("CUDA")
        if torch.cuda.device_count() > 1:
            # models = nn.DataParallel(models)
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    checkpoint = torch.load(best_validate_dir, map_location=device)
#     model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def calculate_acc_agreement(steal_model, victim_model, original_test_data):
    # test metric
    print("testing steal models on original dataset...")
    vic_test_dataset = VictimQueryDataset(original_test_data, 'with_label', args)
    vic_test_loader = DataLoader(dataset=vic_test_dataset,
                             batch_size=1,
                             shuffle=False)
    steal_test_dataset = StealQueryDataset(original_test_data, 'with_label', args)
    steal_test_loader = DataLoader(dataset=steal_test_dataset,
                                 batch_size=1,
                                 shuffle=False)
    steal_model.eval()
    victim_model.eval()
    test_acc = []
    v_acc = []
    steal_test_label = to(torch.zeros(0))
    victim_test_label = to(torch.zeros(0))
    # loop = tqdm(enumerate(test_loader), total=len(test_loader), ncols=100)
    with torch.no_grad():
        # for step, test_data in loop:
        for step, test_data in enumerate(vic_test_loader):
            test_text = test_data[0]
            test_labels = to(test_data[1])
            input_ids = to(test_text['input_ids'].squeeze(1))
            attention_mask = to(test_text['attention_mask'].squeeze(1))
            if args.victim_model_version == 'bert_base_uncased':
                token_type_ids = to(test_text['token_type_ids'].squeeze(1))
            else:
                token_type_ids = None
            output = victim_model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  train_labels=test_labels)
            logits = output[1]
            logits = torch.softmax(logits, 1)
            _, victim_test_argmax = torch.max(logits, 1)
            victim_test_label = torch.cat((victim_test_label, victim_test_argmax.squeeze().unsqueeze(0)), 0)

            vacc = (test_labels == victim_test_argmax.squeeze()).float().mean()
            v_acc.append(vacc.item())
            # loop.set_description('Testing on original dataset')
            # loop.set_postfix(test_acc=accuracy.item())

        for step, test_data in enumerate(steal_test_loader):
            test_text = test_data[0]
            test_labels = to(test_data[1])
            input_ids = to(test_text['input_ids'].squeeze(1))
            attention_mask = to(test_text['attention_mask'].squeeze(1))
            if args.steal_model_version == 'bert_base_uncased':
                token_type_ids = to(test_text['token_type_ids'].squeeze(1))
            else:
                token_type_ids = None
            output, pooler_output = steal_model(input_ids=input_ids,
                                                token_type_ids=token_type_ids,
                                                attention_mask=attention_mask)
            logits = output.logits
            logits = torch.softmax(logits, 1)
            _, steal_test_argmax = torch.max(logits, 1)
            steal_test_label = torch.cat((steal_test_label, steal_test_argmax.squeeze().unsqueeze(0)), 0)
            accuracy = (test_labels == steal_test_argmax.squeeze()).float().mean()
            test_acc.append(accuracy.item())

    global agreement
    agreement = (steal_test_label == victim_test_label).float().mean()
    global acc
    acc = np.mean(test_acc)
    global vac
    vac = np.mean(v_acc)
    print("seed: ", seed)
    print("Evaluation Result on %s: victim acc %.4f, steal acc %.4f, agreement %.4f"
          % (args.task_name, vac, acc, agreement))
    agreement = float(agreement.cpu().data.numpy())
    vac = round(vac, 4)
    acc = round(acc, 4)
    agreement = round(agreement, 4)
    ls = "victim acc " + str(vac) + ", steal acc " + str(acc) + ", agreement " + str(agreement)
    with open(output_file, 'w') as file:
        file.writelines(ls)
    if seed % 10 == 0:
        with open(all_result, 'a') as file:
            file.writelines('----------query_num=' + str(args.query_num) + '------------%\n')
    with open(all_result, 'a') as file:
        file.writelines(str(acc) + ", " + str(agreement) + '\n')
    if seed % 10 == 9:
        acc = []
        agg = []
        with open(all_result, 'r') as f:
            l = f.readlines()
            flag = False
            cnt = 0
            for i, x in enumerate(l):
                if flag is False and 'query_num=' + str(args.query_num) in x:
                    flag = True
                    continue
                if flag is True and cnt < 10:
                    acc.append(float(x.split(', ')[0]))
                    agg.append(float(x.split(', ')[1][:-1]))
                    cnt = cnt + 1
                if cnt >= 10:
                    break
        argmax = np.argmax(acc)
        max_data = str(acc[argmax]) + ', ' + str(agg[argmax])
        mean_acc = np.round(np.mean(acc), 4)
        mean_agg = np.round(np.mean(agg), 4)
        std_acc = np.round(np.std(acc), 4)
        std_agg = np.round(np.std(agg), 4)
        mean_std_data = str(mean_acc) + '\pm' + str(std_acc) + ', ' + str(mean_agg) + '\pm' + str(std_agg)
        with open(all_result, 'a') as f:
            f.writelines(max_data + '\n')
            f.writelines(mean_std_data + '\n')
            f.writelines('%------------------------------------------\n')


def main():
    data = read_query_update(args)
    query = data['sentence']
    predict_label = data['label']
    if args.do_data_class_balance:
        query, predict_label = do_banlace(query, predict_label, args.num_labels)
        print("oversample after")

    # label count
    label_num_cnt = []
    for iii in range(args.num_labels):
        label_num_cnt.append(0)
    predict_label = data['label']
    for i in predict_label:
        for j in range(args.num_labels):
            if i == j:
                label_num_cnt[j] = label_num_cnt[j] + 1
                break
    for i, j in enumerate(label_num_cnt):
        print("%d label count: %d" % (i, j))

    # weighted ce
    if args.weighted_cross_entropy:
        global weights
        weights = torch.tensor(label_num_cnt, dtype=torch.float32)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        if torch.cuda.is_available():
            weights = to(weights)
        print(weights)

    data = {
        'sentence': query,
        'label': predict_label
    }

    steal_train, steal_val = split_steal_data(data)

    if args.task_name == 'SST-2':
        test_data = SST2DataProcessor(args).load_data("test")
    elif args.task_name == 'IMDB':
        test_data = IMDBDataProcessor(args).load_data("test")
    elif args.task_name == 'AGNEWS':
        test_data = AGNEWSDataProcessor(args).load_data("test")
    elif args.task_name == 'HATESPEECH':
        test_data = HATESPEECHDataProcessor(args).load_data("test")

    print("vic model: ", args.victim_model_version)
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
    vck = args.victim_model_checkpoint
    if vck[0] != '-':
        vck = '-' + vck
    checkpoint = torch.load(args.saved_model_path + args.task_name + vck, map_location=device)
#     victim_model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
    victim_model.load_state_dict(checkpoint)
    victim_model.eval()
    steal_model = train_steal_model(steal_train, steal_val)
    calculate_acc_agreement(steal_model, victim_model, test_data)

    os.remove(best_validate_dir)


if __name__ == "__main__":
    setup_seed(seed)
    main()

