from torch.utils.data import Dataset
import torch
import logging
import json


logger = logging.getLogger(__name__)


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
                # self.sentences.append(self.process_sent(sentence, [entity1, entity2]))
                if eval(label) == 1:
                    self.labels.append(1)
                elif eval(label) == -1:
                    self.labels.append(0)

                self.entities.append([entity1, entity2])

        logger.info("Number of Example in {}: {}".format(path, str(len(self.labels))))
    
    # def process_sent(self, sentence, entities):
    #     sentence = sentence.replace(entities[0], ' @ {} @ '.format(entities[0]))
    #     sentence = sentence.replace(entities[1], ' @ {} @ '.format(entities[1]))

        return sentence
    
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

                sentence = self.process_target_sentence(sentence, [position1, position2])
                
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