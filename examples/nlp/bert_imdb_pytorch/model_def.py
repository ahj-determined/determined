from typing import Dict, Sequence, Union
import torch
import torch.nn as nn

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler
import data
import constants

import transformers
import nlp
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers.data.processors.squad import SquadResult
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class BertIMDBPyTorch(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        self.context = context

        self.model = self.context.wrap_model(
                transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased"))

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.optimizer = self.context.wrap_optimizer(
         torch.optim.SGD(
            self.model.parameters(),
            lr=self.context.get_hparam("learning_rate"),
            momentum=0.9)) 

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
            ds = nlp.load_dataset('imdb', split=f"{split}[:1%]")
            ds = ds.map(_tokenize, batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.train_ds, self.test_ds = map(_prepare_ds, ('train', 'test'))

    def build_training_data_loader(self):
        return DataLoader(self.train_ds, drop_last=True, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self):
        return DataLoader(self.test_ds, drop_last=True, batch_size=self.context.get_per_slot_batch_size(),)

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        logits, = self.model(input_ids, mask)
        return logits

    def train_batch(self, batch, epoch_idx, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)
        return {"loss": loss}

    def evaluate_batch(self, batch):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()
        return {"validation_accuracy": acc}

