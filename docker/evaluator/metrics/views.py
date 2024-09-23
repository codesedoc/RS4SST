import threading
import traceback
from django.shortcuts import render
from evaluate import EvaluationModule
# Create your views here.
import json
import logging
import os
from django.middleware.csrf import get_token
from django.http import HttpResponse, HttpRequest
from django.views.decorators.csrf import requires_csrf_token, csrf_protect, csrf_exempt

from . import NAME_METRIC_DICT

def extract_metric_and_input(request: HttpRequest):
	metric_input = json.loads(request.body.decode())
	name = metric_input.pop("name")
	metric = NAME_METRIC_DICT[name]
	return metric, metric_input

@csrf_exempt
def features(request: HttpRequest):
	try:
		metric, metric_input = extract_metric_and_input(request)
		result = metric.info.features.to_dict()
	except Exception as e:
		return HttpResponse(json.dumps({"error": str(e), "tracebacke": traceback.format_exc()}), status=401)
	return HttpResponse(json.dumps(result))

METRIC_LOCK = threading.Lock()

@csrf_exempt
def hf_evaluator(request: HttpRequest):
	try:
		metric, metric_input = extract_metric_and_input(request)
		with METRIC_LOCK:
			result = metric.compute(**metric_input)
	except Exception as e:
		return HttpResponse(json.dumps({"error": str(e), "tracebacke": traceback.format_exc()}), status=401)
		# raise e
	return HttpResponse(json.dumps(result))

# logger = logging.getLogger()
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(logging.Formatter("[%(asctime)s - %(name)s - %(levelname)s] %(message)s"))
# logger.addHandler(console_handler)
# logger.error(metric_intput)