
import torch.nn as nn
import torch
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, BertModel
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, RobertaModel
from transformers import XLNetTokenizer, XLNetConfig, XLNetForSequenceClassification, XLNetModel


class BFSC(nn.Module):
    def __init__(self, args):
        super(BFSC, self).__init__()
        bert_hidden_size = args.bert_hidden_size
        config = BertConfig.from_pretrained(
            args.victim_bert_path,
            num_labels=args.num_labels,
            hidden_dropout_prob=args.bert_hidden_dropout_prob,
            output_hidden_states=args.bert_output_hidden_states
        )
        self.bert = BertForSequenceClassification.from_pretrained(args.victim_bert_path, config=config)

    def forward(self, input_ids, token_type_ids, attention_mask, train_labels):
        output = self.bert(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask,
                           labels=train_labels)
        return output


class RFSC(nn.Module):
    def __init__(self, args):
        super(RFSC, self).__init__()
        # config = RobertaConfig.from_pretrained(
        #     args.victim_roberta_path,
        #     num_labels=args.num_labels,
        #     hidden_dropout_prob=args.roberta_hidden_dropout_prob,
        #     output_hidden_states=args.roberta_output_hidden_states
        # )
        # config = RobertaConfig.from_pretrained(
        #     args.victim_roberta_path,
        #     num_labels=args.num_labels)
        # self.roberta = RobertaForSequenceClassification.from_pretrained(args.victim_roberta_path, config=config)
        self.roberta = RobertaForSequenceClassification.from_pretrained(args.victim_roberta_path, num_labels=args.num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask, train_labels):
        output = self.roberta(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=train_labels)
        return output


class XFSC(nn.Module):
    def __init__(self, args):
        super(XFSC, self).__init__()
        xlnet_hidden_size = args.xlnet_hidden_size
        config = XLNetConfig.from_pretrained(
            args.victim_xlnet_path,
            num_labels=args.num_labels,
            hidden_dropout_prob=args.xlnet_hidden_dropout_prob,
            output_hidden_states=args.xlnet_output_hidden_states
        )
        self.xlnet = XLNetForSequenceClassification.from_pretrained(args.victim_xlnet_path, config=config)

    def forward(self, input_ids, token_type_ids, attention_mask, train_labels):
        output = self.xlnet(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask,
                           labels=train_labels)
        return output