import os
import random
import numpy as np
from tqdm import tqdm

import sys

sys.path.append('../..')
from MeaeQ.utils.config import *
from MeaeQ.utils.tools import *
from MeaeQ.utils.DataProcessor import *
from MeaeQ.utils.MyDataset import *
from MeaeQ.models.victim_models import *
args = ArgParser().get_parser()

import torch
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, BertModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids = [0, 1]
vac = 0.0
seed = 30
weights = []

def main():
    if args.task_name == 'SST-2':
        train_data, validate_data = SST2DataProcessor(args).load_data("train")
        test_data = SST2DataProcessor(args).load_data("test")
    elif args.task_name == 'IMDB':
        train_data, validate_data = IMDBDataProcessor(args).load_data("train")
        test_data = IMDBDataProcessor(args).load_data("test")
    elif args.task_name == 'AGNEWS':
        train_data, validate_data = AGNEWSDataProcessor(args).load_data("train")
        test_data = AGNEWSDataProcessor(args).load_data("test")
    elif args.task_name == 'HATESPEECH':
        train_data, validate_data = HATESPEECHDataProcessor(args).load_data("train")
        test_data = HATESPEECHDataProcessor(args).load_data("test")

    label_num_cnt = []
    for iii in range(args.num_labels):
        label_num_cnt.append(0)
    predict_label = train_data['label']
    for i in predict_label:
        for j in range(args.num_labels):
            if i == j:
                label_num_cnt[j] = label_num_cnt[j] + 1
                break
    for i, j in enumerate(label_num_cnt):
        print("%d label count: %d" % (i, j))

    if args.weighted_cross_entropy:
        global weights
        weights = torch.tensor(label_num_cnt, dtype=torch.float32)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        if torch.cuda.is_available():
            weights = to(weights)
        print(weights)

    train_dataset = VictimQueryDataset(train_data, 'with_label', args)
    validate_dataset = VictimQueryDataset(validate_data, 'with_label', args)
    test_dataset = VictimQueryDataset(test_data, 'with_label', args)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False)

    print('building models')
    print(args.victim_model_version)
    if args.victim_model_version == 'bert_base_uncased':
        model = BFSC(args)
    elif args.victim_model_version == 'roberta_base':
        model = RFSC(args)
    elif args.victim_model_version == 'xlnet_base':
        model = XFSC(args)
    if torch.cuda.is_available():
        print("CUDA")
        if torch.cuda.device_count() > 1:
            # models = nn.DataParallel(models)
            model = torch.nn.DataParallel(model, device_ids=device_ids)
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
    if args.scheduler == 'Step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.8, last_epoch=-1)
    elif args.scheduler == 'MultiStep':
        scheduler = MultiStepLR(optimizer, milestones=[10, 30, 80], gamma=0.8, last_epoch=-1)
    elif args.scheduler == 'Exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.5, last_epoch=-1)
    elif args.scheduler == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.1 * args.learning_rate, last_epoch=-1)
    elif args.scheduler == 'self-definition':
        scheduler = None

    # loss
    if args.weighted_cross_entropy:
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    print("loader size " + str(len(train_loader)))

    print('training models')

    if not os.path.exists(args.saved_model_path):
        os.mkdir(args.saved_model_path)
    train_loss = []
    train_acc = []
    vali_loss = []
    vali_acc = []
    best_vacc = -1
    Epochs = args.num_epochs
    for epoch in range(Epochs):
        if scheduler is None:
            p = float(epoch) / Epochs
            lr = args.learning_rate / (1. + 10 * p) ** 0.75
            optimizer.lr = lr

        train_cost_vector = []
        train_acc_vector = []
        model.train()

        loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        for step, train_data in loop:
        # for step, train_data in enumerate(train_loader):
            train_text = train_data[0]
            train_labels = to(train_data[1])
            input_ids = to(train_text['input_ids'].squeeze(1))
            token_type_ids = to(train_text['token_type_ids'].squeeze(1))
            attention_mask = to(train_text['attention_mask'].squeeze(1))
            #             print(step, "--- input shape", input_ids.shape)
            #             print(step, "--- label shape", train_labels.shape)
            # logits, pooler_output = models(input_ids, token_type_ids, attention_mask, train_labels)
            #             train_labels = train_labels.unsqueeze(-1)
            #             print('label',train_labels.shape)
            #             print('input id',input_ids.shape)
            output = model(input_ids, token_type_ids, attention_mask, train_labels)
            optimizer.zero_grad()

            class_loss = output[0].mean()
            logits = output[1]

            # class_loss = criterion(logits, train_labels)
            #             print("label:", train_labels.dtype)
            #             print("class loss:", class_loss)
            #             print("logits:", logits)
#             print(class_loss.shape)
#             print(class_loss)
            class_loss.backward()
            optimizer.step()
            logits = torch.softmax(logits, 1)
            _, argmax = torch.max(logits, 1)

            accuracy = (train_labels == argmax.squeeze()).float().mean()

            train_cost_vector.append(class_loss.item())
            train_acc_vector.append(accuracy.item())

            loop.set_description('Epoch [%d/%d]' % (epoch + 1, Epochs))
            loop.set_postfix(loss=class_loss.item(),
                             train_acc=accuracy.item())
        tloss = np.mean(train_cost_vector)
        tacc = np.mean(train_acc_vector)
        train_loss.append(tloss)
        train_acc.append(tacc)
        if scheduler is not None:
            scheduler.step()
        validate_cost_vector = []
        validate_acc_vector = []
        model.eval()

        for validate_data in validate_loader:
            validate_text = validate_data[0]
            validate_labels = to(validate_data[1])
            input_ids = to(validate_text['input_ids'].squeeze(1))
            token_type_ids = to(validate_text['token_type_ids'].squeeze(1))
            attention_mask = to(validate_text['attention_mask'].squeeze(1))

            output = model(input_ids, token_type_ids, attention_mask, validate_labels)
            logits = output[1]
            logits = torch.softmax(logits, 1)
            _, argmax = torch.max(logits, 1)
            accuracy = (validate_labels == argmax.squeeze()).float().mean()

            #             validate_cost_vector.append(class_loss.item())
            validate_acc_vector.append(accuracy.item())

        #         vloss = np.mean(validate_cost_vector)
        vacc = np.mean(validate_acc_vector)
        #         vali_loss.append(vloss)
        vali_acc.append(vacc)
        print('Epoch [%d/%d],  train_loss: %.4f, train_acc: %.4f, vali_acc: %.4f'
              % (epoch + 1, Epochs, tloss, tacc, vacc))
        if vacc > best_vacc:
            best_vacc = vacc
            best_validate_dir = args.saved_model_path + args.task_name + args.victim_model_checkpoint
            torch.save(model.state_dict(), best_validate_dir)

    print('testing models')

    best_validate_dir = args.saved_model_path + args.task_name + args.victim_model_checkpoint
    model = BFSC(args)
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

    test_acc = []
    data_features = None
    data_labels = None
    loop = tqdm(enumerate(test_loader), total=len(test_loader), ncols=100)
    # for step, test_data in enumerate(test_loader):
    for step, test_data in loop:
        test_text = test_data[0]
        test_labels = to(test_data[1])
        input_ids = to(test_text['input_ids'].squeeze(1))
        token_type_ids = to(test_text['token_type_ids'].squeeze(1))
        attention_mask = to(test_text['attention_mask'].squeeze(1))
        output = model(input_ids, token_type_ids, attention_mask, test_labels)
        logits = output[1]
        logits = torch.softmax(logits, 1)
        _, test_argmax = torch.max(logits, 1)

        accuracy = (test_labels == test_argmax.squeeze()).float().mean()
        test_acc.append(accuracy.item())
        loop.set_description('Testing ')
        loop.set_postfix(test_acc=accuracy.item())

    acc = np.mean(test_acc)
    global vac
    vac = acc
    print("test acc", acc)



if __name__ == "__main__":
    setup_seed(seed)
    main()

