from typing import List, Callable

from .prompt import Prompt, EXAMPLE_DELIMITER
from .synthesis import SynthesisPositivePrompt, SynthesisNegativePrompt
from .reduction import ReductionPositivePrompt, ReductionNegativePrompt
from .self_refine_tst import Pos2NegPrompt, Neg2PosPrompt

import re
from utils import TransferMode, PromptPhrase, Approach, Task




_MAP_PHRASE_TO_PROMPT = {
    PromptPhrase.REDUCTION: {
        TransferMode.NEG2POS: ReductionNegativePrompt,
        TransferMode.POS2NEG: ReductionPositivePrompt
    },
    PromptPhrase.SYNTHESIS: {
        TransferMode.NEG2POS: SynthesisPositivePrompt,
        TransferMode.POS2NEG: SynthesisNegativePrompt,
    },
    PromptPhrase.SELF_REFINE: {
        TransferMode.NEG2POS: Neg2PosPrompt,
        TransferMode.POS2NEG: Pos2NegPrompt
    }
}

_MAP_APPROACH_TO_PHRASES = {
    Approach.SELF_REFINE_TST: (PromptPhrase.SELF_REFINE,),
    Approach.REDUCTION_SYNTHESIS: (PromptPhrase.REDUCTION, PromptPhrase.SYNTHESIS)
}


def get_prompts(approach: Approach, transfer_mode: TransferMode, task: Task) -> List[Prompt]:
    phrases = _MAP_APPROACH_TO_PHRASES[approach]

    return [_MAP_PHRASE_TO_PROMPT[p][transfer_mode](task) for p in phrases]


def stop_condition(feedback: str, call_back: Callable = True, **call_back_kwargs: dict):
    result = False
    if re.match(r'\s*[Yy]es', feedback):
        result = True
    else:
        if isinstance(call_back, Callable):
            result = call_back(**call_back_kwargs)
    return result
