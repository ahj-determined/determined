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
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        self.config_class, self.tokenizer_class, self.model_class = constants.MODEL_CLASSES[
            self.context.get_hparam("model_type")
        ]
        self.tokenizer = self.tokenizer_class.from_pretrained(
            self.context.get_data_config().get("pretrained_model_name"),
            do_lower_case=True,
            cache_dir=None
        )

        cache_dir_per_rank = f"/tmp/{self.context.distributed.get_rank()}"

        config = self.config_class.from_pretrained(
            self.context.get_data_config().get("pretrained_model_name"),
            cache_dir=cache_dir_per_rank,
        )

        self.model = self.context.wrap_model(
                transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased"))
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

        '''
        self.model = self.context.wrap_model(self.model_class.from_pretrained(
            self.context.get_data_config().get("pretrained_model_name"),
            from_tf=bool(".ckpt" in self.context.get_data_config().get("pretrained_model_name")),
            config=config,
            cache_dir=cache_dir_per_rank,
        ))

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.context.get_hparam("weight_decay"),
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = self.context.wrap_optimizer(AdamW(
            optimizer_grouped_parameters,
            lr=self.context.get_hparam("learning_rate"),
            eps=self.context.get_hparam("adam_epsilon")
        ))
        '''

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

    '''
    def build_training_data_loader(self):
        train_dataset, _, _ = data.load_and_cache_examples(
            data_dir=self.download_directory,
            tokenizer=self.tokenizer,
            task=self.context.get_data_config().get("task"),
            max_seq_length=self.context.get_hparam("max_seq_length"),
            doc_stride=self.context.get_hparam("doc_stride"),
            max_query_length=self.context.get_hparam("max_query_length"),
            evaluate=False,
        )
        return DataLoader(train_dataset, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self):
        self.validation_dataset, self.validation_examples, self.validation_features = data.load_and_cache_examples(
            data_dir=self.download_directory,
            tokenizer=self.tokenizer,
            task=self.context.get_data_config().get("task"),
            max_seq_length=self.context.get_hparam("max_seq_length"),
            doc_stride=self.context.get_hparam("doc_stride"),
            max_query_length=self.context.get_hparam("max_query_length"),
            evaluate=True,
        )
        return DataLoader(
            self.validation_dataset,
            batch_size=self.context.get_per_slot_batch_size(),
        )

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }
        outputs = self.model(**inputs)
        loss = outputs[0]

        self.context.backward(loss)
        self.context.step_optimizer(
            self.optimizer,
            clip_grads=lambda params: torch.nn.utils.clip_grad_norm_(
                params, self.context.get_hparam("max_grad_norm")
            )
        )

        return {"loss": loss}

    def evaluate_full_dataset(self, data_loader: DataLoader):
        all_results = []
        for batch in data_loader:
            inputs = {
                "input_ids": batch[0].cuda(),
                "attention_mask": batch[1].cuda(),
                "token_type_ids": batch[2].cuda(),
            }
            feature_indices = batch[3]
            outputs = self.model(**inputs)
            for i, feature_index in enumerate(feature_indices):
                eval_feature = self.validation_features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                output = [output[i].detach().cpu().tolist() for output in outputs]
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

        output_prediction_file = None
        output_nbest_file = None
        output_null_log_odds_file = None
        predictions = compute_predictions_logits(
            self.validation_examples,
            self.validation_features,
            all_results,
            self.context.get_hparam("n_best_size"),
            self.context.get_hparam("max_answer_length"),
            True,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            True,
            False,
            self.context.get_hparam("null_score_diff_threshold"),
            self.tokenizer,
        )
        results = squad_evaluate(self.validation_examples, predictions)
        return results
    '''
