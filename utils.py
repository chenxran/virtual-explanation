import os
import random
import subprocess
from collections import Counter, defaultdict
import json

import numpy as np
import torch
from datasets import DatasetDict, Dataset


def construct_virtual_explanation(exp_num=10, num_explanation_tokens=4):
    texts = []
    for i in range(exp_num):
        a = ' '.join(["[explanation{}]".format(j) for j in range(3 * i * num_explanation_tokens, (3 * i + 1) * num_explanation_tokens)])
        # b = ' '.join(["[explanation{}]".format(j) for j in range((3 * i + 1) * num_explanation_tokens, (3 * i + 2) * num_explanation_tokens)])
        # c = ' '.join(["[explanation{}]".format(j) for j in range((3 * i + 2) * num_explanation_tokens, (3 * i + 3) * num_explanation_tokens)])
        # texts.append(' <mask> '.join([a, b, c]))
        texts.append('<mask> {} <mask>'.format(a))
    return texts


def evaluate_tacred(true_labels, predicted_labels, label2id):
    NO_RELATION = label2id['NA']
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == NO_RELATION and predicted_label == NO_RELATION:
            pass
        elif true_label == NO_RELATION and predicted_label != NO_RELATION:
            guessed_by_relation[predicted_label] += 1
        elif true_label != NO_RELATION and predicted_label == NO_RELATION:
            gold_by_relation[true_label] += 1
        elif true_label != NO_RELATION and predicted_label != NO_RELATION:
            guessed_by_relation[predicted_label] += 1
            gold_by_relation[true_label] += 1
            if true_label == predicted_label:
                correct_by_relation[predicted_label] += 1
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro


def print_config(logger, config):
    config = vars(config)
    logger.info("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (32 - len(key)))
        logger.info("{} -->   {}".format(keystr, val))
    logger.info("**************** MODEL CONFIGURATION ****************")


def load_explanation(task):
    if task == 'disease':
        path = 'data/explanation/disease.txt'
    elif task == 'spouse':
        path = 'data/explanation/spouse.txt'
    elif task == 'tacred':
        path = 'data/explanation/tacred.txt'

    with open(path, 'r') as f:
        explanations = f.readlines()
        for i, exp in enumerate(explanations):
            explanations[i] = exp.strip()
        
    return explanations


def replace_exp_with_random_tokens(explanations, tokenizer, base_tokenizer_length, replace_ratio, replace_with_new_token):
    for i, exp in enumerate(explanations):
        ids = torch.tensor(tokenizer.convert_tokens_to_ids(exp))
        probability_matrix = torch.full(ids.shape, replace_ratio)
        random_indices = torch.bernoulli(probability_matrix).bool()

        if not replace_with_new_token:
            random_words = torch.randint(base_tokenizer_length, size=ids.size()).long()
        else:
            random_words = torch.randint(base_tokenizer_length, len(tokenizer), size=ids.size()).long()
        
        ids[random_indices] = random_words[random_indices]
        explanations[i] = tokenizer.decode(ids)
    
    return explanations