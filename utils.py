from copy import deepcopy
import json
import logging
import os.path
import typing
from abc import abstractclassmethod, abstractmethod
from dataclasses import dataclass, fields
from enum import StrEnum, auto, Enum
from typing import Dict, Iterable, Mapping, List
import datasets


def get_logger(name="tmp", dump_path="./tmp.log", formatter = None) -> logging.Logger:
    assert isinstance(name, str) and len(name) > 0
    if not (isinstance(dump_path, str) and len(dump_path) > 0):
        dump_path = "./tmp.log"
    
    logger = logging.getLogger(name)

    dump_path_dir = os.path.dirname(dump_path)
    if not os.path.isdir(dump_path_dir):
        os.makedirs(dump_path_dir, exist_ok=True)

    formatter = formatter if isinstance(formatter, logging.Formatter) else logging.Formatter("[%(asctime)s - %(name)s - %(levelname)s] %(message)s")

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(dump_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def read_lines(file_name, mode='r'):
    assert os.path.isfile(file_name)
    with open(file_name, mode=mode) as f:
        result = [l.strip() for l in f.readlines()]
    return result


class Sentiment(StrEnum):
    POSITIVE = auto()
    NEUTRAL = auto()
    NEGATIVE = auto()


class TransferMode(StrEnum):
    NEG2POS = auto()
    POS2NEG = auto()

    @property
    def target(self):
        return {self.NEG2POS: Sentiment.POSITIVE, self.POS2NEG: Sentiment.NEGATIVE}[self]


class PromptPhrase(StrEnum):
    REDUCTION = auto()
    SYNTHESIS = auto()
    SELF_REFINE = auto()


class Task(StrEnum):
    YELP = auto()
    AMAZON = auto()


NUMERIC_CLASS = (int, float)


def add_name2value(total, single):
    for name, value in single.items():
        if value is not None:
            itme = total.get(name, None)
            if isinstance(value, NUMERIC_CLASS):
                if itme is None:
                    itme = 0
                itme += value
            else:
                assert isinstance(value, list)
                if itme is None:
                    itme = []
                itme.extend(value)
            total[name] = itme
    return total


def average_name2value(total, length):
    if length <= 0:
        return
    for name, value in total.items():
        if isinstance(value, NUMERIC_CLASS):
            total[name] /= length
    return total


def sum_name2value(items):
    assert isinstance(items, typing.Sequence)
    assert len(items) > 0
    assert isinstance(items[0], Mapping)
    total = items[0]
    for i in items[1:]:
     total = add_name2value(total, i)
    total = average_name2value(total, length=len(items))
    return total


class EvaluationOutput:
    @staticmethod
    def _add_name2value(total, single):
        def _add(value1, value2):
            if type(value1) != type(value2):
                return None
            if isinstance(value1, NUMERIC_CLASS):
                return value1 + value2
            elif isinstance(value1, list):
                assert len(value1) == len(value2) 
                return [_add(v1,v2) for v1, v2 in zip(value1, value2)]
            elif isinstance(value1, dict):
                assert len(set(value1.keys())-set(value2.keys())) == 0
                return {n: _add(value1[n], value2[n]) for n in value1}
            else:
                return None
            
        total = _add(total, single)
        return total

    @staticmethod
    def _average_name2value(total, length):
        if length <= 0:
            return None
        if isinstance(total, NUMERIC_CLASS):
            return total/length
        elif isinstance(total, list):
            return [EvaluationOutput._average_name2value(v, length) for v in total]
        elif isinstance(total, dict):
            return {n: EvaluationOutput._average_name2value(total[n], length) for n in total}
        else:
            return None

    @classmethod
    @abstractmethod
    def from_dict(cls, evaluation_dict):
        pass

    @abstractmethod
    def to_dict(self):
        pass


class AutoEvaluationOutput(EvaluationOutput):
    def __init__(self):
        self._metric_dict = {}

    @staticmethod
    def sum(evaluations):
        add_name2value = EvaluationOutput._add_name2value
        average_name2value = EvaluationOutput._average_name2value

        assert len(evaluations) >= 1

        result = evaluations[0].to_dict()

        for ev in evaluations[1:]:
            assert isinstance(ev, AutoEvaluationOutput)
            result = add_name2value(result, ev.to_dict())

        result = AutoEvaluationOutput.from_dict(average_name2value(result, length=len(evaluations)))
        return result

    def __str__(self):
        return str(self.to_dict())

    @classmethod
    def from_dict(cls, evaluation_dict):
        result = cls()
        result._metric_dict = evaluation_dict
        return result

    def to_dict(self):
        return deepcopy(self._metric_dict)


@dataclass
class HumanEvaluationOutput(EvaluationOutput):
    content: float = None
    style: float = None
    fluency: float = None

    @staticmethod
    def sum(evaluations):
        add_name2value = EvaluationOutput._add_name2value
        average_name2value = EvaluationOutput._average_name2value
        name2value = {f.name: f.default for f in fields(HumanEvaluationOutput)}
        for ev in evaluations:
            assert isinstance(ev, HumanEvaluationOutput)
            ev = {f.name: ev.__getattribute__(f.name) for f in fields(HumanEvaluationOutput)}
            name2value = add_name2value(name2value, ev)
        average_name2value(name2value, length=len(evaluations))
        result = HumanEvaluationOutput.from_dict(name2value)
        return result

    @classmethod
    def from_dict(cls, evaluation_dict):
        for v in evaluation_dict.values():
            if v is not None:
                assert 0 <= v <= 5
        result = cls(**evaluation_dict)
        return result

    def to_dict(self):
        return dict(content = self.content, style = self.style, fluency = self.fluency)


class Approach(StrEnum):
    REDUCTION_SYNTHESIS = auto()
    SELF_REFINE_TST = auto()

    @property
    def origin_sentiment(self):
        return _MAP_FROM_TRANSFER_MODE_TO_ORIGIN_AND_TARGET_SENTIMENT[self][0]

    @property
    def target_sentiment(self):
        return _MAP_FROM_TRANSFER_MODE_TO_ORIGIN_AND_TARGET_SENTIMENT[self][1]


_MAP_FROM_TRANSFER_MODE_TO_ORIGIN_AND_TARGET_SENTIMENT = {
    TransferMode.POS2NEG: (Sentiment.POSITIVE, Sentiment.NEGATIVE),
    TransferMode.NEG2POS: (Sentiment.NEGATIVE, Sentiment.POSITIVE),
}


def transfer_mode2target_style(mode: TransferMode):
    return _MAP_FROM_TRANSFER_MODE_TO_ORIGIN_AND_TARGET_SENTIMENT[mode][1]


def transfer_mode2source_style(mode: TransferMode):
    return _MAP_FROM_TRANSFER_MODE_TO_ORIGIN_AND_TARGET_SENTIMENT[mode][1]


def load_dataset(dataset_dir) -> Dict[str, datasets.Dataset]:
    assert isinstance(dataset_dir, str) and len(dataset_dir) > 0
    result = dict()
    _MAP_FROM_TRANSFER_MODE_TO_DATA_FILE_NAME = {
        TransferMode.POS2NEG: ("gt_positive_input.txt", "gt_positive_output.txt"),
        TransferMode.NEG2POS: ("gt_negative_input.txt", "gt_negative_output.txt"),
    }

    def _load_dataset(file_names) -> Dict[str, datasets.Dataset]:
        input_lines = read_lines(file_names[0])
        reference_lines = read_lines(file_names[1])
        example_num = len(input_lines)
        assert example_num == len(reference_lines)
        for i, r in zip(input_lines, reference_lines):
            assert len(i.strip()) >0
            assert len(r.strip()) > 0

        return datasets.Dataset.from_dict({
            "index": [i for i in range(example_num)],
            "input": input_lines,
            "reference": reference_lines,
            PromptPhrase.REDUCTION: [None] * example_num,
            PromptPhrase.SYNTHESIS: [None] * example_num
        })

    result[TransferMode.POS2NEG] = _load_dataset([dataset_dir + "/" + i for i in _MAP_FROM_TRANSFER_MODE_TO_DATA_FILE_NAME[TransferMode.POS2NEG]])
    result[TransferMode.NEG2POS] = _load_dataset([dataset_dir + "/" + i for i in _MAP_FROM_TRANSFER_MODE_TO_DATA_FILE_NAME[TransferMode.NEG2POS]])
    return result

import requests

def request(run, url, *args, **kwargs) -> str:
    try:
        body = None
        result = run(url, *args, **kwargs)
        body = result.content.decode(result.encoding)
        assert result.status_code//100 == 2
        data = json.loads(body)
        return data
    except Exception as e:
        raise ValueError(f"Request Failed: {e}!!\nURL: {url}\nargs: {args}\nkwargs: {kwargs}\nresponse: {body}")

class API:
    def __init__(self, host:str, port:int, protocal = "http") -> None:
        assert isinstance(host, str) and len(host) > 0
        assert port > 0 and port < 65536
        self.host = host
        self.port = port
        self.protocal = protocal
        
    @property
    def url_prefix(self):
        return f"{self.protocal}://{self.host}:{self.port}"

    @property
    def name(self) -> str:
        raise NotImplementedError

class EvaluatorAPI(API):
    def __init__(self, *args,  metric:str = "tst", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metric = metric

    @property
    def name(self) -> str:
        return "evaluator"
    
    def autometric(self, predictions, references=None, inputs=None, target_styles=None):
        return request(requests.post, f"{self.url_prefix}/metrics/hf_evaluator",json={
                    "predictions": predictions,
                    "references": references,
                    'inputs': inputs,
                    'sentiment_labels': target_styles,
                    "name": self.metric}
                )
    
class ModelAPI(API):
    def __init__(self, host: str, port: str) -> None:
        super().__init__(host, port)
        self._models = None

class OllamaAPI(ModelAPI):

    @property
    def name(self) -> str:
        return "ollama"
    
    @property
    def models(self) -> str:
        if self._models is None:
            self._models = {i["name"]: i for i in request(requests.get, f"{self.url_prefix}/api/tags")["models"]}
            
        return self._models
    
    def pull_model(self, model):
        if model not in self.models:
            get_logger().info(f"pull model {model}")
            request(requests.post, f"{self.url_prefix}/api/pull", json={"name": model, "stream": False})

    def generate(self, model:str, prompt:str, options:dict ={}) -> str:
        assert isinstance(model, str) and len(model) > 0
        assert isinstance(prompt, str) and len(prompt) > 0
        post_data = dict()
        url = f"{self.url_prefix}/api/generate"
        # post_data["raw"] = True
        post_data["model"] = model
        post_data["prompt"] = prompt
        post_data["stream"] = False
        post_data["options"] = options
        result = request(requests.post, url, json=post_data)
        if not result["done"]:
            get_logger().error(result)
            raise ValueError
        text = result["response"]
        return text
