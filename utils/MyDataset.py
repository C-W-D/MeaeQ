import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer, GPT2Tokenizer


class StealQueryDataset(Dataset):
    def __init__(self, dataset, flag, args):
        self.tokenize_max_length = args.tokenize_max_length
        self.task = args.task_name
        self.steal_model_version = args.steal_model_version
        if args.steal_model_version == 'bert_base_uncased':
            self.tokenizer = BertTokenizer.from_pretrained(args.steal_bert_vocab_path,
                                                           do_lower_case=args.do_lower_case)
        elif args.steal_model_version == 'roberta_base':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            # self.tokenizer = RobertaTokenizer.from_pretrained(args.steal_roberta_vocab_path)
        elif args.steal_model_version == 'roberta_large':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        elif args.steal_model_version == 'xlnet_base':
            self.tokenizer = XLNetTokenizer.from_pretrained(args.steal_xlnet_vocab_path)
        elif args.steal_model_version == 'gpt2_medium':
            # self.tokenizer = XLNetTokenizer.from_pretrained(args.steal_xlnet_vocab_path)
            print("gpt2 tokenizer define")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # self.tokenizer.padding_side = "left"
        elif args.steal_model_version == 'gpt2_small':
            # self.tokenizer = XLNetTokenizer.from_pretrained(args.steal_xlnet_vocab_path)
            print("gpt2 tokenizer define")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # self.tokenizer.padding_side = "left"
        if flag == 'with_label':
            self.sentence = dataset['sentence']
            self.label = torch.from_numpy(np.array(dataset['label']))
        elif flag == 'without_label':
            self.sentence = dataset
            self.label = torch.zeros(len(self.sentence))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sentence = self.sentence[idx]

        if self.steal_model_version == 'gpt2_medium':
            encoded_pair = self.tokenizer(sentence,
                                            padding='max_length',
                                            truncation=True,
                                            add_special_tokens=True,
                                            max_length=self.tokenize_max_length,
                                            return_tensors='pt')
        else:
            encoded_pair = self.tokenizer(sentence,
                                            padding='max_length',
                                            truncation=True,
                                            add_special_tokens=True,
                                            max_length=self.tokenize_max_length,
                                            return_tensors='pt')

        return encoded_pair, self.label[idx]


class VictimQueryDataset(Dataset):
    def __init__(self, dataset, flag, args):
        self.tokenize_max_length = args.tokenize_max_length
        self.task = args.task_name
        if args.victim_model_version == 'bert_base_uncased':
            self.tokenizer = BertTokenizer.from_pretrained(args.victim_bert_vocab_path,
                                                           do_lower_case=args.do_lower_case)
        elif args.victim_model_version == 'roberta_base':
            self.tokenizer = RobertaTokenizer.from_pretrained(args.victim_roberta_vocab_path)
        elif args.victim_model_version == 'xlnet_base':
            self.tokenizer = XLNetTokenizer.from_pretrained(args.victim_xlnet_vocab_path)
        if args.task_name == 'MNLI' or args.task_name == 'SNLI':
            if flag == 'with_label':
                self.sentence = dataset['sentence_pair']
                self.label = torch.from_numpy(np.array(dataset['label']))
            elif flag == 'without_label':
                self.sentence = dataset['sentence_pair']
                self.label = torch.zeros(len(self.sentence))

        else:
            if flag == 'with_label':
                self.sentence = dataset['sentence']
                self.label = torch.from_numpy(np.array(dataset['label']))
            elif flag == 'without_label':
                self.sentence = dataset
                self.label = torch.zeros(len(self.sentence))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        sentence = self.sentence[idx]

        encoded_pair = self.tokenizer(sentence,
                                        padding='max_length',
                                        truncation=True,
                                        add_special_tokens=True,
                                        max_length=self.tokenize_max_length,
                                        return_tensors='pt')

        return encoded_pair, self.label[idx]