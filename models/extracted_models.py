import torch.nn as nn
import torch
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, BertModel
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, RobertaModel
from transformers import XLNetTokenizer, XLNetConfig, XLNetForSequenceClassification, XLNetModel
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW, GPT2Config


class BPC(nn.Module):
    def __init__(self, args):
        super(BPC, self).__init__()
        config = BertConfig.from_pretrained(
            args.steal_bert_path,
            num_labels=args.num_labels,
            hidden_dropout_prob=args.bert_hidden_dropout_prob,
            output_hidden_states=args.bert_output_hidden_states
        )
        self.bert = BertForSequenceClassification.from_pretrained(args.steal_bert_path, config=config)

    def forward(self, input_ids, token_type_ids, attention_mask, train_labels=None):
        if train_labels is None:
            with torch.no_grad():
                output = self.bert(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)
                pooler_output = self.bert.bert(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask).pooler_output
        else:
            output = self.bert(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               labels=train_labels)
            pooler_output = self.bert.bert(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask).pooler_output
        return output, pooler_output


class RPC(nn.Module):
    def __init__(self, args):
        super(RPC, self).__init__()
        # config = RobertaConfig.from_pretrained(
        #     args.steal_roberta_path,
        #     num_labels=args.num_labels,
        #     output_hidden_states=True
        # )
        # self.roberta = RobertaForSequenceClassification.from_pretrained(args.steal_roberta_path, config=config)

        name = None
        if args.steal_model_version == 'roberta_base':
            name = 'roberta-base'
        elif args.steal_model_version == 'roberta_large':
            name = 'roberta-large'
        config = RobertaConfig.from_pretrained(
            name,
            num_labels=args.num_labels,
            output_hidden_states=True
        )
        self.roberta = RobertaForSequenceClassification.from_pretrained(name, config=config)

    def forward(self, input_ids, token_type_ids, attention_mask, train_labels=None):
        if train_labels is None:
            with torch.no_grad():
                output = self.roberta(input_ids=input_ids,
                                      token_type_ids=token_type_ids,
                                      attention_mask=attention_mask)
                pooler_output = self.roberta.roberta(input_ids=input_ids,
                                               token_type_ids=token_type_ids,
                                               attention_mask=attention_mask).pooler_output
        else:
            output = self.roberta(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  labels=train_labels)
            pooler_output = self.roberta.roberta(input_ids=input_ids,
                                           token_type_ids=token_type_ids,
                                           attention_mask=attention_mask).pooler_output
        return output, pooler_output


class XPC(nn.Module):
    def __init__(self, args):
        super(XPC, self).__init__()
        config = XLNetConfig.from_pretrained(
            args.steal_xlnet_path,
            num_labels=args.num_labels,
            output_hidden_states=True
        )
        self.xlnet = XLNetForSequenceClassification.from_pretrained(args.steal_xlnet_path, config=config)

    def forward(self, input_ids, token_type_ids, attention_mask, train_labels=None):
        if train_labels is None:
            with torch.no_grad():
                output = self.xlnet(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                      attention_mask=attention_mask)
                pooler_output = output.hidden_states[-1][0][0].unsqueeze(0)
        else:
            output = self.xlnet(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  labels=train_labels)
            # pooler_output = self.xlnet.base_model(input_ids=input_ids,
            #                                      token_type_ids=token_type_ids,
            #                                      attention_mask=attention_mask).pooler_output
            pooler_output = output.hidden_states[-1][0][0].unsqueeze(0)
        return output, pooler_output


class GPT2PC(nn.Module):
    def __init__(self, args):
        super(GPT2PC, self).__init__()
        # config = GPT2Config.from_pretrained(
        #     args.steal_gpt2_path,
        #     num_labels=args.num_labels,
        #     output_hidden_states=True
        # )
        name = None

        if args.steal_model_version == 'gpt2_small':
            name = 'gpt2'
        elif args.steal_model_version == 'gpt2_medium':
            name = 'gpt2-medium'
        self.tokenizer = GPT2Tokenizer.from_pretrained(name)
        config = GPT2Config.from_pretrained(
            name,
            num_labels=args.num_labels,
            output_hidden_states=True
        )
        self.gpt2 = GPT2ForSequenceClassification.from_pretrained(name, config=config)
        self.gpt2.config.pad_token_id = self.tokenizer.eos_token_id

    def forward(self, input_ids, token_type_ids, attention_mask, train_labels=None):
        if train_labels is None:
            with torch.no_grad():
                output = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        else:
            output = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, labels=train_labels)
        return output, output.hidden_states[-1][:, -1, :]
