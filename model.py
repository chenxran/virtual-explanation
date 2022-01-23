from torch import nn
from transformers import (AutoConfig, AutoModel, BertConfig, BertTokenizer, BertModel,
                          AutoModelForSequenceClassification, AutoTokenizer)


class ExpBERT(nn.Module):
    def __init__(self, args, exp_num):
        super(ExpBERT, self).__init__()
        self.args = args
        self.exp_num = exp_num
        self.config = AutoConfig.from_pretrained(args.model)
        self.model = AutoModel.from_pretrained(args.model, config=self.config)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.config.hidden_size * exp_num, args.num_labels),
        )

        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        ):
        
        pooler_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        ).pooler_output

        pooler_output = pooler_output.reshape(len(labels), self.args.exp_num * self.config.hidden_size).contiguous()
        logits = self.classifier(pooler_output)

        loss = self.criterion(logits, labels)

        return {
            "loss": loss, 
            "logits": logits,
        }

    def resize_token_embeddings(self, length):
        self.model.resize_token_embeddings(length)


class DenseExpBERT(nn.Module):
    def __init__(self, args, exp_num):
        super(ExpBERT, self).__init__()
        self.args = args
        self.exp_num = exp_num
        self.config = AutoConfig.from_pretrained(args.model)
        self.model = AutoModel.from_pretrained(args.model, config=self.config)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.config.hidden_size * exp_num, args.num_labels),
        )

        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        ):
        
        pooler_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        ).pooler_output

        pooler_output = pooler_output.reshape(len(labels), self.args.exp_num * self.config.hidden_size).contiguous()
        logits = self.classifier(pooler_output)

        loss = self.criterion(logits, labels)

        return {
            "loss": loss, 
            "logits": logits,
        }

    def resize_token_embeddings(self, length):
        self.model.resize_token_embeddings(length)