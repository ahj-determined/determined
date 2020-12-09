#!/usr/bin/env python3

#from absl import app, flags, logging

#import sh

import torch as th
#import pytorch_lightning as pl

import nlp
import transformers

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext

class IMDBSentimentClassifier(PyTorchTrial):
    def __init__(self, context):
        self.context = context
        self.model = self.context.wrap_model(transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased"))
        self.loss = th.nn.CrossEntropyLoss(reduction='none')
        self.optimizer = self.context.wrap_optimizer(th.optim.SGD(
            self.model.parameters(),
            lr=self.context.get_hparam("learning_rate"),
            momentum=0.9,
        ))

        self.prepare_data()

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

        def _tokenize(x):
            x['input_ids'] = tokenizer.batch_encode_plus(
                    x['text'], 
                    max_length=32, 
                    pad_to_max_length=True)['input_ids']
            return x

        def _prepare_ds(split):
            ds = nlp.load_dataset('imdb', split=f'{split}[:100%]')
            ds = ds.map(_tokenize, batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.train_ds, self.test_ds = map(_prepare_ds, ('train', 'test'))

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        logits, = self.model(input_ids, mask)
        return logits

    def train_batch(self, batch, epoch_idx, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)
        return {'loss': loss}

    def evaluate_batch(self, batch):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()
        return {'validation_acc': acc}

    def build_training_data_loader(self):
        return DataLoader(
                self.train_ds,
                batch_size=self.context.get_per_slot_batch_size(),
                drop_last=True,
                )

    def build_validation_data_loader(self):
        return DataLoader(
                self.test_ds,
                batch_size=self.context.get_per_slot_batch_size(),
                drop_last=True,
                )

