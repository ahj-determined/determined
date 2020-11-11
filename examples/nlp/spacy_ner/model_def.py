import random
from typing import Any, Dict
import warnings
import spacy
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer
from spacy.gold import GoldParse

from determined.python import PythonTrial, PythonTrialContext

# TODO: better NER data
TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
    ("Where is Idaho?", {"entities": [(9, 14, "LOC")]}),
    ("My favorite city is Boston.", {"entities": [(20, 26, "LOC")]})
]

VAL_DATA = [
    ("Who is Gene Simmons?", [(7, 19, "PERSON")]),
    ("Is New York City your favorite place?",  [(3, 16, "LOC")])
]

class SpacyNerTrial(PythonTrial):
    def __init__(self, context: PythonTrialContext) -> None:
        self.context = context

        self.model = spacy.blank("en")

        ner = self.model.create_pipe("ner")
        self.model.add_pipe(ner, last=True)

        # for all labels ner.add_label(label)
        ner.add_label('LOC')
        ner.add_label('PERSON')

    def train_some(self) -> Dict[str, Any]:
        # get names of other pipes to disable them during training
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in self.model.pipe_names if pipe not in pipe_exceptions]
        with self.model.disable_pipes(*other_pipes), warnings.catch_warnings():
            self.model.begin_training()
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                self.model.update(
                    texts,
                    annotations,
                    #drop=self.context.get_hparam("dropout"),
                    drop=0.5,
                    losses=losses,
                )
            return {"loss": losses['ner']}

    def evaluate_full_dataset(self) -> Dict[str, Any]:
        scorer = Scorer()
        for input_, annot in VAL_DATA:
            doc_gold_text = self.model.make_doc(input_)
            gold = GoldParse(doc_gold_text, entities=annot)
            pred_value = self.model(input_)
            scorer.score(pred_value, gold)
        scorer.scores
        return {
            "val_precision": scorer.scores['ents_p'],
            "val_recall": scorer.scores['ents_r'],
            "val_f1": scorer.scores['ents_f'],
        }

    def save(self, path: str):
        self.model.to_disk(path)

    def load(self, path: str) -> None:
        self.model = spacy.load(path)
