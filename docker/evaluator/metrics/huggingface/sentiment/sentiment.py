import math

import evaluate
import datasets
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pysentimiento.analyzer import AnalyzerForSequenceClassification, create_analyzer


def acc_compute(predictions, labels):
    return round(np.average(np.array(np.array([predictions]) == np.array([labels]), dtype=float)), 4)


class sentiment(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.ClassLabel(names=["negative", "neutral", "positive"]),
            }),
            description="",
            citation="",
        )

    def _download_and_prepare(self, dl_manager):
        self._classifier: AnalyzerForSequenceClassification = create_analyzer(
            task="sentiment", lang="en")

    def _compute(self, predictions, references=None, **kwargs):
        classifier = self._classifier
        category2id = self.info.features['references'].str2int
        id2category = self.info.features['references'].int2str
        label2category = {
            "NEG": "negative",
            "POS": "positive",
            "NEU": "neutral",
        }
        result = classifier.predict(predictions)
        senti_preds = []
        probabilities = {s: [] for s in label2category.values()}
        for a in result:
            senti_preds.append(label2category[a.output])
            for l, p in a.probas.items():
                probabilities[label2category[l]].append(p)
        for n in probabilities:
            probabilities[n] = round(np.average(np.array(probabilities[n], dtype=float)), 4)
       
        metrics = dict()
        if references is not None:
            labels = np.array([int(s) for s in references])
            acc = acc_compute(np.array(category2id(senti_preds)), labels)
            metrics["acc"] = acc
        metrics["sentiments"] = senti_preds
        metrics["probabilities"] = probabilities
        return metrics