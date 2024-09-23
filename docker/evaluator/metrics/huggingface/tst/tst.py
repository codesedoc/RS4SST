import os
import evaluate
import datasets


class tst(evaluate.Metric):
    def __init__(self, *args, global_name_metric_dict = None, **kwargs):
        self.global_name_metric_dict = global_name_metric_dict
        super().__init__(*args, **kwargs)
    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
                'inputs': datasets.Value('string'),
                'sentiment_labels': self.global_name_metric_dict["sentiment"].info.features["references"],
            }),
            description="",
            citation="",
        )
    
    def _download_and_prepare(self, dl_manager):
        self._sacrebleu = self.global_name_metric_dict["sbleu"]
        self._semantic = self.global_name_metric_dict["semantic"]
        self._sentiment = self.global_name_metric_dict["sentiment"]
        self._ppl = self.global_name_metric_dict["ppl"]
        # print(f"TST: {id(self._ppl)}")
        

    def _compute(self, predictions, references, inputs, sentiment_labels, **kwargs):
        metrics = dict()
        metrics.update({f"r_{k}": v for k, v in self._sacrebleu.compute(predictions=predictions, references=references).items()})
        metrics.update({f"s_{k}": v for k, v in self._sacrebleu.compute(predictions=predictions, references=inputs).items()})
        
        
        metrics.update({f"r_{k}": v for k, v in self._semantic.compute(predictions=predictions, references=references).items()})
        metrics.update({f"s_{k}": v for k, v in self._semantic.compute(predictions=predictions, references=inputs).items()})
        metrics.update(self._ppl.compute(predictions=predictions))
        metrics.update(self._sentiment.compute(predictions=predictions, references=sentiment_labels))
        return metrics