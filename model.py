#-*- coding:utf-8 -*-
import warnings

import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel
from transformers import AutoConfig, AutoModel

warnings.filterwarnings("ignore")


class BertClassifier(nn.Module):
    def __init__(self, bert_path, dropout_rate, hidden_size=768, output_feature_size=256):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, output_feature_size),
            nn.BatchNorm1d(output_feature_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_feature_size, 5)
        )

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]

        _, pooled = self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              output_all_encoded_layers=False)
        context_embedding = self.dropout(pooled)

        output = self.classifier(context_embedding) 
        output = F.softmax(output, dim=1)
        return output


class AutoBertClassifier(nn.Module): # Debert
    def __init__(self, bert_path, dropout_rate=0.5, num_labels=5,
                 bert_hidden_size=768, linear_hidden_size=256):
        super(AutoBertClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(bert_path)
        self.bert = AutoModel.from_config(self.config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size, linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(linear_hidden_size, num_labels)
        )

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]

        _, pooled = self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              # output_hidden_states=False,
                              # return_dict=False
                              )
        context_embedding = self.dropout(pooled)

        output = self.classifier(context_embedding)
        output = F.softmax(output, dim=1)
        return output
