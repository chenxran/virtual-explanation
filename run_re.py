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
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AddedToken

from utils import (construct_virtual_explanation, evaluate_tacred,
                   load_explanation)

from model import ExpBERT
from data import TACREDDataset, REDataset
# torch.set_printoptions(profile="full")


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


class Trainer(object):
    def __init__(self, args, seed, checkpoint=None):
        self.args = args
        self.seed = seed

        explanations = []
        if args.explanation and not args.mannual_exp and not args.mixture:
            explanations = construct_virtual_explanation(1, args.num_explanation_tokens)
        elif args.explanation and args.mannual_exp and not args.mixture:
            explanations = load_explanation(args.task)
        elif args.explanation and not args.mannual_exp and args.mixture:
            if args.task == 'tacred':
                explanations = []
                # for efficiency, we only use 1/3 of the explanations in TACRED
                for i, exp in enumerate(load_explanation(args.task)):
                    if i % 3 == 0:
                        explanations.append(exp)
            else:
                explanations = load_explanation(args.task)
            explanations.extend(construct_virtual_explanation(1, args.num_explanation_tokens))  #  + load_explanation(args.task)
            args.exp_num = len(explanations)
        else:
            pass

        # output explanations used for training
        logger.info('Explanations used for training:')
        for i, exp in enumerate(explanations):
            logger.info(f'{i + 1}. {exp}')

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model, cache_dir='/data/chenxingran/models')
        base_tokenizer_length = len(self.tokenizer)
        self.base_tokenizer_length = base_tokenizer_length

        if args.explanation:
            self.model = ExpBERT(args, len(explanations), base_tokenizer_length).cuda()
        else:
            config = AutoConfig.from_pretrained(args.model, cache_dir='/data/cache/huggingface/models')
            config.num_labels = args.num_labels
            self.model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config, cache_dir='/data/cache/huggingface/models').cuda()

        # add additional tokens to tokenizer & initialize parameters
        # whenever we use virtual exp, we need to add additional tokens
        if args.mixture or (args.explanation and not args.mannual_exp):
            self.tokenizer.add_tokens([f" [explanation{i}]" for i in range(1 * args.exp_num * args.num_explanation_tokens)])
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.model.embeddings.word_embeddings.weight.data[base_tokenizer_length:] = torch.normal(
                mean=-0.028, 
                std=0.0427, 
                size=(len(self.tokenizer) - base_tokenizer_length, self.model.model.embeddings.word_embeddings.weight.shape[1])
            ) 

            # check whether we add tokens of virtual explanation to the tokenizer
            logger.info('Tokenized explanations:')
            for i, exp in enumerate(explanations):
                logger.info(f'{i + 1}. {self.tokenizer.tokenize(exp)}')

        # load dataset
        if args.task == 'tacred':
            with open('data/tacred/label2id.json', 'r', encoding='utf-8') as file:
                label2id = json.load(file)
            self.train_dataset = TACREDDataset(self.args, TASK2PATH[self.args.trainset], explanations, self.tokenizer, label2id)
            self.eval_dataset = TACREDDataset(self.args, TASK2PATH[self.args.testset], explanations, self.tokenizer, label2id)
            self.label2id = label2id
        elif args.task in ['disease', 'spouse']:
            self.train_dataset = REDataset(self.args, TASK2PATH[self.args.trainset], explanations, self.tokenizer)
            self.eval_dataset = REDataset(self.args, TASK2PATH[self.args.testset], explanations, self.tokenizer)

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

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        if self.args.task == "tacred":
            self.train_iterator = self.train_loader
            self.eval_iterator = self.eval_loader
        else:
            self.train_iterator = [examples for examples in self.train_loader]
            self.eval_iterator = [examples for examples in self.eval_loader]

            logger.info('check tokenized data: ')
            for sample_input_ids in self.train_iterator[0].input_ids[:3]:
                logger.info(f'{self.tokenizer.convert_ids_to_tokens(sample_input_ids)}')

    def compute_metrics(self, labels, predictions):
        accuracy = accuracy_score(y_pred=predictions, y_true=labels)

        if args.task == 'tacred':
            _, _, f1 = evaluate_tacred(labels, predictions, self.label2id)
        else:
            f1 = f1_score(y_pred=predictions, y_true=labels, average='macro' if self.args.task == 'tacred' else 'binary')

        return accuracy, f1

    def train(self):
        # use metrics to keep evaluation results
        all_eval_accuracy = []
        all_eval_f1 = []

        # start training
        for e in range(self.args.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            with tqdm(total=len(self.train_iterator)) as pbar:
                for step, examples in enumerate(self.train_iterator):
                    losses = []
                    for k, v in examples.items():
                        examples[k] = v.cuda()
                    outputs = self.model(**examples)
                    (outputs["loss"] / self.args.gradient_accumulation_steps).backward()
                    losses.append(outputs['loss'].item())
                    if step % self.args.gradient_accumulation_steps == 0 or step == len(self.train_loader) - 1:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        if self.args.exp_type == "fixed_random":
                            self.model.model.embeddings.word_embeddings.weight.grad[self.base_tokenizer_length:] = 0.
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    pbar.set_description("Training Loss: {}".format(round(np.mean(losses), 4)))
                    pbar.update(1)
            accuracy, f1 = self.evaluate(e)
            all_eval_accuracy.append(accuracy)
            all_eval_f1.append(f1)

        logger.info(f'Evaluation Result on valid set: Accuracy: {round(np.max(all_eval_accuracy), 4)} | F1-Score: {round(np.max(all_eval_f1), 4)}')

        return np.max(all_eval_f1)

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
                    predictions.extend(torch.argmax(outputs['logits'], dim=1).tolist())
                    labels.extend(examples['labels'].cpu().numpy())

                    pbar.update(1)
        
        accuracy, f1 = self.compute_metrics(predictions, labels)
        logger.info(f"Evaluation Result on valid set in Epoch {epoch}: Accuracy: {round(accuracy, 4)} | F1-Score: {round(f1, 4)}. (Number of Data {len(predictions)})")

        return accuracy, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--place_holder", action='store_true', help="whether to add place holder for target sentence to improve performance.")
    parser.add_argument("--task", type=str, default="tacred")
    # parser.add_argument("--warmup_steps", type=float, default=500)

    # use this argument to control the following arguments
    parser.add_argument('--run', type=str, default=None, required=True)

    parser.add_argument("--mannual_exp", type=bool, default=False, help="whether to use annotated explanations.")
    parser.add_argument("--num_explanation_tokens", type=int, default=None, help="number of virtual tokens in each virtual explanations.")
    parser.add_argument("--exp_num", type=int, default=None, help="number of explanations (including annotated explanation or virtual explanation). If use annotated explanations, this value will be automatically set.")
    parser.add_argument("--explanation", type=bool, default=False, help="whether to use explanation (including annotated explanation or virtual explanation).")
    parser.add_argument("--mixture", type=bool, default=False, help='whether to use both annotated and virtual explanations.')
    parser.add_argument("--exp_type", type=str, default="default")
    parser.add_argument("--factorized_dim", type=int, default=None, help='dimension of factorized random layer.')


    args = parser.parse_args()

    args.trainset = args.task + '-train'
    args.evalset = args.task + '-dev'   # we directly evaluate on test set since we do not search hyper parameters.
    args.testset = args.task + '-test'

    args.num_labels = 42 if args.task == 'tacred' else 2

    if args.task == 'tacred':
        args.shuffle = True

    # sanity check
    # if (args.exp_num > 1 or args.mannual_exp) and not args.explanation:
        # raise ValueError('You should use explanation mode.')
    
    # if args.mannual_exp:
    #     if args.task == 'disease':
    #         args.exp_num = 29
    #     elif args.task == 'spouse':
    #         args.exp_num = 41
    #     elif args.task == 'tacred':
    #         raise ValueError('--mannual_exp should be False when --task == tacred since mannual explanations for tacred were not released by ExpBERT.')
    
    # if args.mixture:
    #     if args.mannual_exp:
    #         raise ValueError('--mannual_exp should be False when --mixture is True.') 
    #     if not args.explanation:
    #         raise ValueError('--explanation should be True when --mixture is True.')

    # if args.exp_type in ["dense", "factorized_dense", "factorized_random", "fixed_random"]:
    #     if args.mannual_exp:
    #         raise ValueError('--mannual_exp should be False when --exp_type == dense or factorized_dense.')
    #     if not args.explanation:
    #         raise ValueError('--explanation should be True when --exp_type == dense or factorized_dense.')
    #     if args.exp_num > 1:
    #         raise ValueError('--exp_num should be 1 when --exp_type == dense or factorized_dense.')
    #     if args.mixture:
    #         raise ValueError('--mixture should be False when --exp_type == dense or factorized_dense.')
    
    # set arguments for different runs.
    if args.run == 'baseline':
        args.mannual_exp = False
        args.num_explanation_tokens = None
        args.exp_num = None
        args.explanation = False
        args.mixture = False
        args.exp_type = "default"
        args.factorized_dim = None
    elif args.run == 'baseline-without-place-holder':
        args.mannual_exp = False
        args.num_explanation_tokens = None
        args.exp_num = None
        args.explanation = False
        args.mixture = False
        args.exp_type = "default"
        args.factorized_dim = None

        args.place_holder = False        
    elif args.run == 'virtual-explanation':
        args.mannual_exp = False
        args.num_explanation_tokens = 4
        args.exp_num = 1
        args.explanation = True
        args.mixture = False
        args.exp_type = "default"
        args.factorized_dim = None
    elif args.run == 'virtual-explanation-custom-exp-num':
        args.mannual_exp = False
        args.num_explanation_tokens = 4
        args.exp_num = args.exp_num  # customize
        args.explanation = True
        args.mixture = False
        args.exp_type = "default"
        args.factorized_dim = None
    elif args.run == 'annotated-explanation':
        args.mannual_exp = True
        args.num_explanation_tokens = None
        if args.task == 'disease':
            args.exp_num = 29
        elif args.task == 'spouse':
            args.exp_num = 41
        elif args.task == 'tacred':
            raise ValueError('--mannual_exp should be False when --task == tacred since mannual explanations for tacred were not released by ExpBERT.')
        args.explanation = True
        args.mixture = False
        args.exp_type = "default"
        args.factorized_dim = None

        # we need to set batch size to 1 in order to prevent gpu memory overflow.
        args.batch_size = 1
        args.gradient_accumulation_steps = 32
    elif args.run == 'mixture':
        args.mannual_exp = False
        args.num_explanation_tokens = 4
        args.exp_num = 1
        args.explanation = True
        args.mixture = True
        args.exp_type = "default"
        args.factorized_dim = None

        # we need to set batch size to 1 in order to prevent gpu memory overflow.
        args.batch_size = 1
        args.gradient_accumulation_steps = 32
    elif args.run == 'fixed-random':
        args.mannual_exp = False
        args.num_explanation_tokens = 4
        args.exp_num = 1
        args.explanation = True
        args.mixture = False
        args.exp_type = "fixed_random"
        args.factorized_dim = None
    elif args.run == 'factorized-random':
        args.mannual_exp = False
        args.num_explanation_tokens = 4
        args.exp_num = 1
        args.explanation = True
        args.mixture = False
        args.exp_type = "factorized_random"
        args.factorized_dim = 8
    elif args.run == 'factorized-dense':
        args.mannual_exp = False
        args.num_explanation_tokens = 4
        args.exp_num = 1
        args.explanation = True
        args.mixture = False
        args.exp_type = "factorized_dense"
        args.factorized_dim = None
    elif args.run == 'dense':
        args.mannual_exp = False
        args.num_explanation_tokens = 4
        args.exp_num = 1
        args.explanation = True
        args.mixture = False
        args.exp_type = "dense"
        args.factorized_dim = None
    else:
        raise ValueError(
            '--run should be one of the following:'
            'baseline, baseline-without-place-holder, virtual-explanation,'
            'virtual-explanation-custom-exp-num, annotated-explanation, mixture,'
            'fixed-random, factorized-random, factorized-dense, dense.'
        )

    logging.basicConfig(
    filename=f'logs2/no-freeze-{args.run}-{str(datetime.now())}.log',
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)

    print_config(args)
    

    wandb.init(project="virtual-explanation", entity='chenxran', config=args)

    results = []
    for i, seed in enumerate(range(42, 47)):
        logger.info(f'********** run {i} with random seed {seed} **********')
        set_random_seed(seed)
        trainer = Trainer(args, seed)
        results.append(trainer.train())
        logger.info(f'******************* run {i} completed! ******************')
    
    logger.info(f'Average F1-score: {round(np.mean(results), 4)}')
    logger.info(f'Std F1-score: {round(np.std(results), 4)}')
    logger.info(f'95% CI: {np.std(results) * 1.96 / np.sqrt(len(results))}')

    wandb.log(
        {
            'F1': round(np.mean(results), 4) * 100,
            'F1_std': round(np.std(results), 4) * 100,
            'F1_95_CI': round(np.std(results) * 1.96 / np.sqrt(len(results)), 4) * 100,
        }
    )
    wandb.finish()