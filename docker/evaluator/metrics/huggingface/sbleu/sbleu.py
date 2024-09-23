import evaluate
import datasets


class sbleu(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            description="",
            citation="",
        )
    
    def _download_and_prepare(self, dl_manager):
        self._sacrebleu = evaluate.load("sacrebleu")

    def _compute(self, predictions = None, references = None, **kwargs):
        metrics = {
            "sbleu": self._sacrebleu.compute(predictions=predictions, references=references)["score"]
        }
        return metrics