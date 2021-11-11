import argparse
import copy
import json
import logging
import math
import os
import random
import re
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel, BertConfig, BertTokenizer, BertModel,
                          AutoModelForSequenceClassification, AutoTokenizer, BertTokenizerFast,
                          BertForSequenceClassification,
                          get_scheduler)

from utils import (evaluate_tacred,
                   load_explanation, replace_exp_with_random_tokens)
torch.set_printoptions(profile="full")

logging.basicConfig(
    filename='logs/freeze-analysis-{}.log'.format(str(datetime.now())),
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


TASK2PATH = {
    "disease-train": "data/disease/train.txt",
    "disease-test": "data/disease/test.txt",
    "spouse-train": "data/spouse/train.txt",
    "spouse-test": "data/spouse/test.txt",
    'tacred-train': "data/tacred/train.json",
    'tacred-dev': "data/tacred/dev.json",
    'tacred-test': "data/tacred/test.json",
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_config(config):
    config = vars(config)
    logger.info("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (32 - len(key)))
        logger.info("{} -->   {}".format(keystr, val))
    logger.info("**************** MODEL CONFIGURATION ****************")


class REDataset(Dataset):
    def __init__(self, args, path, explanations, tokenizer):
        super(REDataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.explanations = explanations
        
        self.sentences = []
        self.labels = []
        self.entities = []

        self.load(path)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as file:
            data = file.readlines()
            for example in data:
                sentence, entity1, entity2, id, label = example.strip().split("\t")
                self.sentences.append(sentence)
                if eval(label) == 1:
                    self.labels.append(1)
                elif eval(label) == -1:
                    self.labels.append(0)

                self.entities.append([entity1, entity2])

        logger.info("Number of Example in {}: {}".format(path, str(len(self.labels))))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return {
            "sentence": self.sentences[index],
            "entity": self.entities[index],
            'labels': self.labels[index],
        }
    
    def collate_fn(self, batch):
        labels = []
        sentence1 = []
        sentence2 = []
        if self.args.explanation:
            for ex in batch:
                for exp in self.explanations:
                    exp = self.insert_entity(exp, ex['entity'])
                    sentence1.append(ex['sentence'])
                    sentence2.append(exp)
                labels.append(ex['labels'])

            outputs = self.tokenizer(
                sentence1, sentence2,
                add_special_tokens=True,
                padding="longest",
                truncation=True,
                max_length=176,
                return_tensors="pt",
            )
        else:
            for ex in batch:
                sentence1.append(ex['sentence'])
                labels.append(ex['labels'])
            
            outputs = self.tokenizer(
                sentence1,
                add_special_tokens=True,
                padding="longest",
                truncation=True,
                max_length=156,
                return_tensors="pt",
            )

        outputs['labels'] = torch.tensor(labels)
        return outputs

    def insert_entity(self, exp, entities):
        if '<mask>' in exp:
            for entity in entities:
                index = exp.index('<mask>')
                exp = exp[:index] + entity + exp[index + len('<mask>'):]
        else:
            exp = exp.replace('{e1}', entities[0])
            exp = exp.replace('{e2}', entities[1])

        return exp


class TACREDDataset(Dataset):
    def __init__(self, args, path, explanations, tokenizer, label2id):
        super(Dataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.explanations = explanations
        
        self.sentences = []
        self.labels = []
        self.entities = []

        self.label2id = label2id

        self.load(path)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            data  = json.load(file)
            for example in data:
                label = example['label']
                sentence = example['text']
                entity1 = example['ents'][0][0]
                position1 = (example['ents'][0][1], example['ents'][0][2])
                entity2 = example['ents'][1][0]
                position2 = (example['ents'][1][1], example['ents'][1][2])

                # sentence = self.process_target_sentence(sentence, [position1, position2])
                
                self.labels.append(self.label2id[label])
                self.sentences.append(sentence)
                self.entities.append([entity1, entity2])

        logger.info("Number of Example in {}: {}".format(path, str(len(self.labels))))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return {
            'sentence': self.sentences[index],
            'entity': self.entities[index],
            'labels': self.labels[index],
        }
    
    def collate_fn(self, batch):
        labels = []
        sentence1 = []
        sentence2 = []
        if self.args.explanation:
            for ex in batch:
                for exp in self.explanations:
                    exp = self.insert_entity(exp, ex['entity'])
                    sentence1.append(ex['sentence'])
                    sentence2.append(exp)
                labels.append(ex['labels'])

            outputs = self.tokenizer(
                sentence1, sentence2,
                add_special_tokens=True,
                padding="longest",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
        else:
            for ex in batch:
                sentence1.append(ex['sentence'])
                labels.append(ex['labels'])
            
            outputs = self.tokenizer(
                sentence1,
                add_special_tokens=True,
                padding="longest",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

        outputs['labels'] = torch.tensor(labels)
        return outputs

    def insert_entity(self, exp, entities):
        if '<mask>' in exp:
            for entity in entities:
                index = exp.index('<mask>')
                exp = exp[:index] + entity + exp[index + len('<mask>'):]
        else:
            exp = exp.replace('{e1}', entities[0])
            exp = exp.replace('{e2}', entities[1])

        return exp

    # def process_target_sentence(self, sentence, positions):
    #     return ' @ '.join([sentence[:positions[0][0]], sentence[positions[0][0]:positions[0][1]], sentence[positions[0][1]:positions[1][0]], sentence[positions[1][0]:positions[1][1]], sentence[positions[1][1]:]])


class Classifier(nn.Module):
    def __init__(self, args, config, exp_num, num_explanation_tokens):
        super(Classifier, self).__init__()
        self.args = args
        self.config = config
        self.exp_num = exp_num

        if args.replace_with_new_token:
            self.embedding = nn.Parameter(torch.normal(mean=-0.028, std=0.0427, size=(num_explanation_tokens, config.hidden_size)))  # (args.exp_num * args.num_explanation_tokens * 3, config.hidden_size))
        
        if args.projection_dim == 0:
            self.projection_dim = config.hidden_size
            self.projection_layer = lambda x: x
        else:
            self.projection_dim = args.projection_dim
            self.projection_layer = nn.Sequential(
                nn.Linear(config.hidden_size, args.projection_dim),
            )
        
        if args.num_layers == 1:
            self.classifier = nn.Sequential(
                nn.Linear(self.projection_dim * exp_num, args.hidden_dim),
                nn.Dropout(p=args.dropout),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.num_labels),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.projection_dim * exp_num, args.num_labels),
            )

        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, encoder=None, fake_exp_id=None):
        if self.args.replace_with_new_token:
            inputs_embeds = encoder.get_input_embeddings()(input_ids)
            inputs_embeds[input_ids == fake_exp_id, :] = self.embedding.repeat(len(labels), 1, 1).view(-1, self.config.hidden_size)
            pooler_output = encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
        else:
            with torch.no_grad():
                pooler_output = encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
        pooler_output = self.projection_layer(pooler_output).reshape(len(labels), self.exp_num * (args.projection_dim if args.projection_dim != 0 else 768)).contiguous()
        logits = self.classifier(pooler_output.cuda())
        loss = self.criterion(logits, labels)

        return {
            "loss": loss, 
            "logits": logits,
        }

    def resize_token_embeddings(self, length):
        self.model.resize_token_embeddings(length)

class Trainer(object):
    def __init__(self, args, seed, checkpoint=None):
        self.args = args
        self.seed = seed
        print_config(args)
        explanations = load_explanation(args.task)

        # if checkpoint given, load checkpoint
        # this is used when running tacred task since we need to evaluate on test dataset.
        self.tokenizer = BertTokenizerFast.from_pretrained(self.args.model)

        base_tokenizer_length = len(self.tokenizer)

        if args.explanation:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ['{e1}', '{e2}', '<exp>']})
            self.fake_exp_id = self.tokenizer.convert_tokens_to_ids('<exp>')
            def replace_exp_with_random_tokens(explanations, tokenizer, base_tokenizer_length, replace_ratio, replace_with_new_token):
                num_explanation_tokens = 0
                for i, exp in enumerate(explanations):
                    ids = torch.tensor(tokenizer(exp, add_special_tokens=False)['input_ids'])

                    probability_matrix = torch.full(ids.shape, replace_ratio)
                    special_tokens_mask = torch.tensor([id in tokenizer.all_special_ids for id in ids])
                    
                    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
                    random_indices = torch.bernoulli(probability_matrix).bool()

                    if not replace_with_new_token:
                        random_words = torch.randint(base_tokenizer_length, size=ids.size()).long()
                    else:
                        random_words = torch.tensor([self.fake_exp_id] * len(ids))  # torch.arange(len(self.tokenizer), len(self.tokenizer) + len(ids))
                        num_explanation_tokens += random_indices.sum()
                    ids[random_indices] = random_words[random_indices]
                    explanations[i] = tokenizer.decode(ids)
                
                return explanations, num_explanation_tokens

            explanations, num_explanation_tokens = replace_exp_with_random_tokens(copy.deepcopy(explanations), self.tokenizer, base_tokenizer_length, args.replace_ratio, args.replace_with_new_token)

        # check explanation
        # with open('check_exp', 'w') as f:
        #     for exp in explanations:
        #         f.write(exp + '\n')

        if args.explanation:
            model_path = 'data/pretrain_models/scibert' if args.task == 'disease' else '/data/chenxingran/explanation/data/pretrain_models/bert'
            config = BertConfig.from_pretrained(model_path)
            self.encoder = BertModel.from_pretrained(model_path, config=config)
            self.encoder.cuda()
            self.encoder.eval()
            self.model = Classifier(args, config, len(explanations), num_explanation_tokens)
            self.model.cuda()
        else:
            config = BertConfig.from_pretrained(args.model)
            config.num_labels = args.num_labels
            model_path = 'data/pretrain_models/scibert' if args.task == 'disease' else 'data/pretrain-models/bert'
            self.model = BertForSequenceClassification.from_pretrained(model_path, config=config).cuda()

        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint))

        self.encoder.resize_token_embeddings(len(self.tokenizer))

        if args.task == 'tacred':
            with open('data/tacred/label2id.json', 'r', encoding='utf-8') as file:
                label2id = json.load(file)
            self.train_dataset = TACREDDataset(self.args, TASK2PATH[self.args.trainset], explanations, self.tokenizer, label2id)
            self.eval_dataset = TACREDDataset(self.args, TASK2PATH[self.args.evalset], explanations, self.tokenizer, label2id)
            self.predict_dataset = TACREDDataset(self.args, TASK2PATH[self.args.testset], explanations, self.tokenizer, label2id)
            self.label2id = label2id
        else:
            self.train_dataset = REDataset(self.args, TASK2PATH[self.args.trainset], explanations, self.tokenizer)
            self.eval_dataset = REDataset(self.args, TASK2PATH[self.args.evalset], explanations, self.tokenizer)
            self.predict_dataset = REDataset(self.args, TASK2PATH[self.args.testset], explanations, self.tokenizer)

        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=args.batch_size, 
            shuffle=args.shuffle, 
            collate_fn=self.train_dataset.collate_fn,
        )
        self.eval_loader = DataLoader(
            self.eval_dataset, 
            batch_size=args.batch_size, 
            shuffle=args.shuffle, 
            collate_fn=self.eval_dataset.collate_fn,
        )

        self.predict_loader = DataLoader(
            self.predict_dataset, 
            batch_size=args.batch_size, 
            shuffle=args.shuffle, 
            collate_fn=self.predict_dataset.collate_fn,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        self.train_iterator = []
        with tqdm(total=len(self.train_loader)) as pbar:
            for examples in self.train_loader:
                self.train_iterator.append(examples)
                pbar.update(1)
                
        self.eval_iterator = []
        with tqdm(total=len(self.eval_loader)) as pbar:
            for examples in self.eval_loader:
                self.eval_iterator.append(examples)
                pbar.update(1)

    def compute_metrics(self, labels, predictions):
        accuracy = accuracy_score(y_pred=predictions, y_true=labels)

        if args.task == 'tacred':
            _, _, f1 = evaluate_tacred(labels, predictions, self.label2id)
        else:
            f1 = f1_score(y_pred=predictions, y_true=labels, average='macro' if self.args.task == 'tacred' else 'binary')

        return accuracy, f1

    def train(self):
        # use metrics to keep evaluation results
        all_accuracy = []
        all_f1 = []

        for e in range(self.args.epochs):
            self.model.train()
            losses = []
            with tqdm(total=len(self.train_iterator)) as pbar:
                self.optimizer.zero_grad()
                for step, examples in enumerate(self.train_iterator):
                    for k, v in examples.items():
                        examples[k] = v.cuda()
                    # check sentence
                    # check_sentence = self.tokenizer.batch_decode(examples['input_ids'])
                    # with open('check_sentence.txt', 'w', encoding='utf-8') as file:
                        # for sent in check_sentence:
                            # file.write(sent + '\n')

                    outputs = self.model(**examples, encoder=self.encoder, fake_exp_id=self.fake_exp_id if args.replace_with_new_token else None)
                    (outputs["loss"] / self.args.gradient_accumulation_steps).backward()
                    losses.append(outputs['loss'].item())
                    if step % self.args.gradient_accumulation_steps == 0 or step == len(self.train_loader) - 1:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                        self.optimizer.step()
                        # self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                    pbar.set_description("Training Loss: {}".format(round(np.mean(losses[-30:]), 4)))
                    pbar.update(1)
            accuracy, f1 = self.evaluate(e)
            all_accuracy.append(accuracy)
            all_f1.append(f1)
            if all_f1[-1] == np.max(all_f1):
                # make dir:
                if not os.path.exists('cache/{}/'.format(args.task)):
                    os.makedirs('cache/{}/'.format(args.task))
                torch.save(self.model.state_dict(), 'cache/{}/best_model_{}.pkl'.format(self.args.task, self.seed))

        logger.info('Evaluation Result on valid set: Accuracy: {} | F1-score: {}'.format(round(np.max(all_accuracy), 4), round(np.max(all_f1), 4)))

    def evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(self.eval_iterator)) as pbar:
                labels = []
                predictions = []
                for step, examples in enumerate(self.eval_iterator):
                    for k, v in examples.items():
                        examples[k] = v.cuda()
                    outputs = self.model(**examples, encoder=self.encoder, fake_exp_id=self.fake_exp_id if args.replace_with_new_token else None)
                    predictions.extend(torch.argmax(outputs['logits'], dim=1).tolist())
                    labels.extend(examples['labels'].cpu().numpy())

                    pbar.update(1)
        
        accuracy, f1 = self.compute_metrics(predictions, labels)
        logger.info("Evaluation Result in Epoch {}: Accuracy: {} | F1-Score: {}. (Number of Data {})".format(epoch, accuracy, f1, len(predictions)))

        return accuracy, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--warmup_steps", type=float, default=500)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--task", type=str, default="tacred")
    
    parser.add_argument("--explanation", type=bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--replace_ratio", type=float, default=0.0)
    parser.add_argument("--replace_with_new_token", type=bool, default=False)

    args = parser.parse_args()

    args.trainset = args.task + '-train'
    args.evalset = args.task + '-test'  # we directly evaluate on test set since we do not search hyper parameters.
    args.testset = args.task + '-test'

    args.num_labels = 42 if args.task == 'tacred' else 2
    # args.shuffle = True if args.task == 'tacred' else False

    # if args.exp_num > 1 and not args.explanation:
    #     raise ValueError('You should use explanation mode.')
    
    if args.task == 'tacred':
        raise ValueError()

    if args.task == 'disease':
        args.model = 'allenai/scibert_scivocab_uncased'

    # repeat experiment five times
    for seed in range(42, 45):
        set_random_seed(seed)
        trainer = Trainer(args, seed)
        trainer.train()
