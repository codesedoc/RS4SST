from abc import abstractmethod
from typing import List

from utils import Task

EXAMPLE_DELIMITER = "###"


class Prompt:
    def __init__(self, task: Task = None):
        assert isinstance(task, Task)
        self.task: Task = task

    @property
    @abstractmethod
    def task2prompt_strs(self) -> List[Task]:
        pass

    @property
    def generation_inference_prefix(self):
        return "Rewrite:"

    @property
    def feedback_inference_prefix(self):
        return "Feedback:"

    @property
    def refine_inference_prefix(self):
        return "Rewrite:"

    @abstractmethod
    def generation_prompt(self, *args, **kwargs):
        pass

    @abstractmethod
    def feedback_prompt(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_prompt(self, *args, **kwargs):
        pass
    #
    # @staticmethod
    # def _prompt_format(prompt_method) -> str:
    #     return prompt_method(
    #         **{k: f'[{k}]' for k, v in inspect.signature(prompt_method).parameters.items() if not k.startswith('*')}
    #     )

    def _prompt_format(self):
        return {
            "generation": self.generation_prompt("[input_text]"),
            "feedback": self.feedback_prompt("[feedback]", "[generation]"),
            "refine": self.refine_prompt("[feedback]", generations=["[generation1]", "[generation1]"],
                                         feed_backs=["[feed_back1]", "[feed_back2]"]),
        }

    @property
    @abstractmethod
    def prompt_format(self):
        pass










