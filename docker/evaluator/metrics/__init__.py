import evaluate
import os
from evaluate import EvaluationModule
import numpy as np
METRICS_DIR = f"{os.path.dirname(__file__)}/huggingface"

NAME_METRIC_DICT = { 
	"ppl": evaluate.load(f"{METRICS_DIR}/ppl"),
	"semantic": evaluate.load(f"{METRICS_DIR}/semantic"),
	"sentiment": evaluate.load(f"{METRICS_DIR}/sentiment"),
	"sbleu": evaluate.load(f"{METRICS_DIR}/sbleu"),
}
NAME_METRIC_DICT["tst"] = evaluate.load(f"{METRICS_DIR}/tst", global_name_metric_dict = NAME_METRIC_DICT)


assert np.all([isinstance(v, EvaluationModule) for v in NAME_METRIC_DICT.values()])