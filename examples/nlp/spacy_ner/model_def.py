import random
from typing import Any, Dict
import warnings
import spacy
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer
from spacy.gold import GoldParse

from determined.python import PythonTrial, PythonTrialContext

import data

class SpacyNerTrial(PythonTrial):
    def __init__(self, context: PythonTrialContext) -> None:
        self.context = context

        self.model = spacy.blank("en")

        ner = self.model.create_pipe("ner")
        self.model.add_pipe(ner, last=True)

        self.training_data, entity_types = data.create_data('train.txt', True)
        self.validation_data, _ = data.create_data('val.txt', False)

        for entity_type in entity_types:
            ner.add_label(entity_type)

        optimizer = self.model.begin_training()
        optimizer.learn_rate = self.context.get_hparam("lr")
        optimizer.beta1 = self.context.get_hparam("beta1")
        optimizer.beta1 = self.context.get_hparam("beta2")

    def train_some(self) -> Dict[str, Any]:
        # get names of other pipes to disable them during training
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in self.model.pipe_names if pipe not in pipe_exceptions]
        with self.model.disable_pipes(*other_pipes), warnings.catch_warnings():
            random.shuffle(self.training_data)
            losses = {}
            batches = minibatch(self.training_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                self.model.update(
                    texts,
                    annotations,
                    drop=self.context.get_hparam("dropout"),
                    losses=losses
                )
            return {"loss": losses['ner']}

    def evaluate_full_dataset(self) -> Dict[str, Any]:
        scorer = Scorer()
        for input_, annot in self.validation_data:
            doc_gold_text = self.model.make_doc(input_)
            gold = GoldParse(doc_gold_text, entities=annot)
            pred_value = self.model(input_)
            scorer.score(pred_value, gold)
        return {
            "val_precision": scorer.scores['ents_p'],
            "val_recall": scorer.scores['ents_r'],
            "val_f1": scorer.scores['ents_f'],
        }

    def save(self, path: str):
        self.model.to_disk(path)

    def load(self, path: str) -> None:
        self.model = spacy.load(path)
        self.model.resume_training()
