import determined as det
from determined.python import PythonTrial
import numpy as np
import pathlib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
import xgboost as xgb

class XGBoostTrial(PythonTrial):
    checkpoint_file = "checkpoint"

    def __init__(self, ctx: det.TrialContext):
        super().__init__(ctx)
        self.train, self.test, self.y_train, self.y_test = self.build_dataloaders()
        self.hparams = ctx.get_hparams()
        self.model, self.model_location = None, None

    def train_some(self):
        if self.model_location:
            self.model = xgb.train(self.hparams, self.train, 10, xgb_model=self.model_location)
        else:
            self.model = xgb.train(self.hparams, self.train, 10)

        preds = self.model.predict(self.train)
        best_preds = np.asarray([np.argmax(line) for line in preds])
        return {
            "precision": precision_score(self.y_train, best_preds, average='macro'),
            "recall": recall_score(self.y_train, best_preds, average='macro'),
            "accuracy": accuracy_score(self.y_train, best_preds),
        }

    def evaluate_full_dataset(self):
        preds = self.model.predict(self.test)
        best_preds = np.asarray([np.argmax(line) for line in preds])
        return {
            "precision": precision_score(self.y_test, best_preds, average='macro'),
            "recall": recall_score(self.y_test, best_preds, average='macro'),
            "accuracy": accuracy_score(self.y_test, best_preds),
        }

    def save(self, path: pathlib.Path):
        self.model.save_model(str(path.joinpath(self.checkpoint_file)))

    def load(self, path: pathlib.Path):
        self.model_location = path.joinpath(self.checkpoint_file)

    @staticmethod
    def build_dataloaders():
        iris = datasets.load_iris()
        x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
        return xgb.DMatrix(x_train, label=y_train), xgb.DMatrix(x_test, label=y_test), y_train, y_test
