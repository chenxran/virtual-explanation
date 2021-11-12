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
                          get_scheduler)

from utils import (construct_virtual_explanation, evaluate_tacred,
                   load_explanation, replace_exp_with_random_tokens)

torch.set_printoptions(profile="full")

logging.basicConfig(
    filename='logs/expbert-{}.log'.format(str(datetime.now())),
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


TASK2PATH = {
    "disease-train": "data/disease/train.txt",
    "disease-dev": "data/disease/dev.txt",
    "disease-test": "data/disease/test.txt",
    "spouse-train": "data/spouse/train.txt",
    "spouse-dev": "data/spouse/dev.txt",
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


class ExpBERT(nn.Module):
    def __init__(self, args, exp_num):
        super(ExpBERT, self).__init__()
        self.args = args
        self.exp_num = exp_num

        # model_path = 'data/pretrain_models/bert' if args.task == 'spouse' else 'data/pretrain_models/scibert'
        self.config = AutoConfig.from_pretrained(args.model)
        self.model = AutoModel.from_pretrained(args.model, config=self.config)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.config.hidden_size * exp_num, self.args.num_labels),
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

        pooler_output = pooler_output.reshape(len(labels), self.exp_num * self.config.hidden_size).contiguous()
        logits = self.classifier(pooler_output)

        loss = self.criterion(logits, labels)

        return {
            "loss": loss, 
            "logits": logits,
        }

    def resize_token_embeddings(self, length):
        self.model.resize_token_embeddings(length)


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
        outputs = {}
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

        # check_sentences1 = self.tokenizer.batch_decode(outputs['input_ids'])
        
        # with open('check_input_sentence.txt', 'w') as f:
        #     for i in range(len(check_sentences1)):
        #         f.write(check_sentences1[i] + '\n')

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
        super(TACREDDataset, self).__init__()
        self.label2id = label2id
        self.args = args
        self.tokenizer = tokenizer
        self.explanations = explanations
        self.sentences = []
        self.labels = []
        self.entities = []
        self.positions = []

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


                self.labels.append(self.label2id[label])
                self.sentences.append(sentence)
                self.entities.append([entity1, entity2])
                self.positions.append([position1, position2])

        logger.info("Number of Example in {}: {}".format(path, str(len(self.labels))))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return {
            'sentence': self.sentences[index],
            'entity': self.entities[index],
            'labels': self.labels[index],
            'position': self.positions[index],
        }
    
    def collate_fn(self, batch):
        if self.args.explanation:
            sentence1 = []
            sentence2 = []
            label = []
            for ex in batch:
                for exp in self.explanations:
                    exp = self.insert_entity(exp, ex['entity'])
                    # sentence1.append(ex['sentence'])
                    sentence1.append(self.process_target_sentence(ex['sentence'], ex['position']))
                    sentence2.append(exp)
                label.append(ex['labels'])

            outputs = self.tokenizer(
                sentence1, sentence2,
                add_special_tokens=True,
                padding="longest",
                truncation=True,
                max_length=128, # 256,
                return_tensors="pt",
            )
            outputs['labels'] = torch.tensor(label)
        else:
            sentence = []
            labels = []
            for ex in batch:
                sentence.append(self.process_target_sentence(ex['sentence'], ex['position']))
                labels.append(ex['labels'])
            
            outputs = self.tokenizer(
                sentence,
                add_special_tokens=True,
                padding="longest",
                truncation=True,
                max_length=128, # 256,
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
    
    def process_target_sentence(self, sentence, positions):
        return ' @ '.join([sentence[:positions[0][0]], sentence[positions[0][0]:positions[0][1]], sentence[positions[0][1]:positions[1][0]], sentence[positions[1][0]:positions[1][1]], sentence[positions[1][1]:]])



class Trainer(object):
    def __init__(self, args, seed, checkpoint=None):
        self.args = args
        self.seed = seed 
        print_config(args)

        # if args.task == 'tacred':  # TODO: delete
        #     explanations = []
        #     for i, exp in enumerate(load_explanation(args.task)):
        #         if i % 3 == 0:
        #             explanations.append(exp)
        # else:
        explanations = load_explanation(args.task)

        # if checkpoint given, load checkpoint
        # this is used when running tacred task since we need to evaluate on test dataset.
        if args.explanation:
            self.model = ExpBERT(args, len(explanations)).cuda()
        else:
            config = AutoConfig.from_pretrained(args.model)
            config.num_labels = args.num_labels
            self.model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).cuda()

        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint))

        self.tokenizer = BertTokenizerFast.from_pretrained(self.args.model)

        base_tokenizer_length = len(self.tokenizer)
        if args.explanation:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ['{e1}', '{e2}']})
            def replace_exp_with_random_tokens(explanations, tokenizer, base_tokenizer_length, replace_ratio, replace_with_new_token):
                for i, exp in enumerate(explanations):
                    ids = torch.tensor(tokenizer(exp, add_special_tokens=False)['input_ids'])

                    probability_matrix = torch.full(ids.shape, replace_ratio)
                    special_tokens_mask = torch.tensor([id in tokenizer.all_special_ids for id in ids])
                    
                    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
                    random_indices = torch.bernoulli(probability_matrix).bool()

                    if not replace_with_new_token:
                        random_words = torch.randint(base_tokenizer_length, size=ids.size()).long()
                    else:
                        random_words = torch.arange(len(self.tokenizer), len(self.tokenizer) + len(ids))
                        self.tokenizer.add_tokens(["[explanation{}_{}]".format(i, j) for j in range(len(random_words))])

                    ids[random_indices] = random_words[random_indices]
                    explanations[i] = tokenizer.decode(ids)
                
                return explanations

            explanations = replace_exp_with_random_tokens(copy.deepcopy(explanations), self.tokenizer, base_tokenizer_length, args.replace_ratio, args.replace_with_new_token)

            # with open('check.txt', 'w') as f:
            #     for i, exp in enumerate(masked_explanations):
            #         f.write(exp + '\n')
            #         f.write(explanations[i] + '\n')

            # self.model.resize_token_embeddings(len(self.tokenizer))

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
            # Subset(self.train_dataset, list(range(500))),  # TODO: need to be changed # self.train_dataset,
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

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        self.train_iterator = self.train_loader  # [examples for examples in self.train_loader]
        self.eval_iterator = self.eval_loader  # [examples for examples in self.eval_loader]

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

        # accuracy, f1 = self.evaluate(-1)
        # all_accuracy.append(accuracy)
        # all_f1.append(f1)

        # start training
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
                    # with open('check_sentence_tacred.txt', 'w') as f:
                    #     for i, check in enumerate(check_sentence):
                    #         f.write(check + '\n')

                    outputs = self.model(**examples)
                    # print(outputs['loss'])
                    (outputs["loss"] / self.args.gradient_accumulation_steps).backward()
                    losses.append(outputs['loss'].item())
                    if step % self.args.gradient_accumulation_steps == 0 or step == len(self.train_loader) - 1:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        # self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                    pbar.set_description("Training Loss: {}".format(round(np.mean(losses[-30:]), 4)))
                    pbar.update(1)

            accuracy, f1 = self.evaluate(e)
            all_accuracy.append(accuracy)
            all_f1.append(f1)
            # if all_f1[-1] == np.max(all_f1):
                # make dir:
                # if not os.path.exists('cache/{}/'.format(args.task)):
                    # os.makedirs('cache/{}/'.format(args.task))
                # torch.save(self.model.state_dict(), 'cache/{}/best_model_{}.pkl'.format(self.args.task, self.seed))

        logger.info('Evaluation Result on valid set: Accuracy: {} | F1-score: {}'.format(round(np.max(all_accuracy), 4), round(np.max(all_f1), 4)))



        # predict on test set
        # try:
        #     self.model.load_state_dict(torch.load('cache/{}/best_model_{}.pkl'.format(self.args.task, self.seed)))
        # except:
        #     raise ValueError('No model found!')

        # self.model.eval()
        # predictions = []
        # labels = []
        # with torch.no_grad():
        #     for step, examples in enumerate(self.predict_loader):
        #         for k, v in examples.items():
        #             examples[k] = v.cuda()
        #         outputs = self.model(**examples)
        #         predictions.extend(torch.argmax(outputs['logits'], dim=1).cpu().numpy())
        #         labels.extend(examples['labels'].cpu().numpy())
        
        # accuracy, f1 = self.compute_metrics(labels, predictions)
        # logger.info('Evaluation Result on test set: Accuracy: {} | F1-Score: {}.'.format(round(accuracy, 4), round(f1, 4)))

    def evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(self.eval_iterator)) as pbar:
                labels = []
                predictions = []
                for step, examples in enumerate(self.eval_iterator):
                    for k, v in examples.items():
                        examples[k] = v.cuda()
                    outputs = self.model(**examples)
                    predictions.extend(torch.argmax(outputs['logits'], dim=1).cpu().numpy())
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
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--warmup_steps", type=float, default=500)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--task", type=str, default="tacred")
    
    parser.add_argument("--replace_ratio", type=float, default=0)
    parser.add_argument("--replace_with_new_token", type=bool, default=False)
    parser.add_argument("--exp_num", type=int, default=1)
    parser.add_argument("--explanation", type=bool, default=False)


    args = parser.parse_args()

    args.trainset = args.task + '-train'
    args.evalset = args.task + '-test'
    args.testset = args.task + '-test'

    args.num_labels = 42 if args.task == 'tacred' else 2

    if (args.exp_num > 1 or args.replace_ratio > 0) and not args.explanation:
        raise ValueError('You should use explanation mode.')

    if args.task == 'tacred':
        args.shuffle = True
    # repeat experiment five times
    for seed in range(12, 17):
        set_random_seed(seed)
        trainer = Trainer(args, seed)
        trainer.train()
