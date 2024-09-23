import evaluate
import numpy as np
import datasets
from sentence_transformers import SentenceTransformer, util


class semantic(evaluate.Metric):
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
		self._model = SentenceTransformer("all-MiniLM-L6-v2")

	def _compute(self, predictions=None, references = None, **kwargs):
        # Sentences are encoded by calling model.encode()
		model = self._model
		assert len(predictions) == len(references)
		similarities = []
		for s1, s2 in zip(predictions, references):
			cos_sim = util.cos_sim(model.encode(s1), model.encode(s2))
			similarities.append(cos_sim.item())
		metrics = {
			"avg_similarity": round(np.average(np.array(similarities)), 4),
			"similarities": list(map(lambda x: round(x, 4), similarities))
		}
		return metrics