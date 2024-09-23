# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import argparse
import collections
import datetime
import json
import logging
import os.path
import threading
import time
import socket
import traceback
from enum import IntEnum, auto, StrEnum
import re
import datasets
import requests
from typing import List, Callable
from prompt import EXAMPLE_DELIMITER, stop_condition as psc, SynthesisPositivePrompt, Neg2PosPrompt, ReductionPositivePrompt, \
    SynthesisNegativePrompt, Pos2NegPrompt, get_prompts
from prompt import Prompt, ReductionNegativePrompt
from utils import API, EvaluatorAPI, OllamaAPI, Sentiment, get_logger, load_dataset, TransferMode, Task, Approach, HumanEvaluationOutput, AutoEvaluationOutput, \
    transfer_mode2target_style


def extract_inference(generation:str, prefix):
    assert isinstance(generation, str) and len(generation) > 0
    assert isinstance(prefix, str) and len(prefix) > 0
    result = None
    for line in generation.split("\n"):
        if len(line) > 0:
            line = line.strip()
            if line.startswith(f"{prefix}"):
                result = line[len(prefix):].strip()
                break
            items = line.split(":")
            if len(items) == 2:
                result = items[-1].strip()
                break

    if not (isinstance(result, str) and len(result) > 0):
        logger.error(generation)
        raise ValueError("Incorrect Inference")
    return result


def generate(prompt_str, inference_prefix):
    logger.debug("**************Prompt**************")
    logger.debug(prompt_str)
    generation = generator.generate(model=args.model, prompt=prompt_str, 
                                    options=dict(temperature=0, top_p=args.top_p, num_predict=48, seed=args.seed))
    inference = extract_inference(generation, inference_prefix)
    logger.debug(f"Inference: {inference}")
    return inference


class StopEvent(StrEnum):
    TRIAL_OVER = auto()
    MEET_CONDITION = auto()


def self_refine(input_text,  prompt: Prompt, stop_condition: Callable, max_iteration: int = 2, reduction_text: str = None, target_style = None):
    logger.info("init_generating")
    init_generation = generate(prompt.generation_prompt(input_text=input_text, reduction_text=reduction_text), inference_prefix=prompt.generation_inference_prefix)
    generations = [init_generation]
    feedbacks = []
    feedback_num = 1
    stop_event = StopEvent.TRIAL_OVER
    assert max_iteration >= feedback_num
    stop_condition_call_back = None
    if target_style is not None and (args.task is Task.AMAZON or Task.YELP):
        def stop_condition_call_back(generation):
            evaluator = EvaluatorAPI(host= args.e_host, port=args.e_port, metric="sentiment")
            senti_result = evaluator.autometric([generation], references=[str(target_style)])
            logger.debug(f"Generation: {generation}")
            return senti_result["sentiments"][0] == target_style

    while feedback_num <= max_iteration:
        logger.info(f"the {feedback_num}-th feedback")
        feedback = generate(prompt.feedback_prompt(input_text=input_text, generation = generations[-1], reduction_text=reduction_text), inference_prefix=prompt.feedback_inference_prefix)
        feedbacks.append(feedback)

        if stop_condition(feedback, stop_condition_call_back, generation = generations[-1]):
            stop_event = StopEvent.MEET_CONDITION
            break
        logger.info(f"the {feedback_num}-th refinement")
        refinement = generate(prompt.refine_prompt(input_text = input_text, generations = generations, feed_backs = feedbacks, reduction_text=reduction_text), inference_prefix=prompt.refine_inference_prefix)
        generations.append(refinement)
        feedback_num += 1
    return {
        "output": generations[-1],
        "feedback_num": min(feedback_num, max_iteration),
        "stop_event": stop_event.value,
        "feedbacks": feedbacks,
        "generations": generations[:-1]
    }


def init_result(prompt):
    # logger.debug(prompt.prompt_format)
    return {
        "metadata": {"start_date": datetime.datetime.now().astimezone().isoformat(timespec='seconds')},
        "examples": [],
        "prompt_format": prompt.prompt_format
    }


def load_result(dump_file):
    with open(dump_file, mode="r") as f:
        result = json.load(f)
    return result


def dump_result(result, dump_file):
    def run():
        file_dir = os.path.dirname(dump_file)
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir, exist_ok=True)
        if os.path.isfile(dump_file):
            backup_file = dump_file + '.backup'
            os.rename(dump_file, backup_file)
            logger.info(f"The old result is at '{backup_file}'.")
        with open(dump_file, mode="w") as f:
            json.dump(result, f, indent=4)
        logger.info(f"Dump the result at '{dump_file}")
    run()
    # dump_thread = threading.Thread(target=run)
    # dump_thread.start()


def tst():
    logger.setLevel(logging.DEBUG)
    if args.online_test:
        while True:
            input_text = input("Enter your source text ('q' for quite): ").strip()
            if input_text == "q":
                break
            if args.approach is Approach.SELF_REFINE_TST:
                pass
            elif args.approach is Approach.REDUCTION_SYNTHESIS:
                rs_online(input_text)
    else:
        prompts = get_prompts(args.approach, args.transfer_mode, args.task)
        inferred_result = None
        inference_factory = None
        if args.approach is Approach.SELF_REFINE_TST:
            assert len(prompts) == 1
            inferred_result = self_refine_tst(dataset, prompts[0])
            inference_factory = lambda example: example["self_refine"]["output"]

        elif args.approach is Approach.REDUCTION_SYNTHESIS:
            assert len(prompts) == 2
            inferred_result = reduction_synthesis(dataset, prompts[0], prompts[1])
            inference_factory = lambda example: example["synthesis"]["output"]

        evaluate(inferred_result, inference_factory = inference_factory)

    
from collections import OrderedDict


def evaluate(inferred_dict, inference_factory: Callable):
    dump_file = os.path.join(args.output_dir, f"{args.approach}-evaluate.txt")
    try:
        if os.path.isfile(dump_file):
            evaluate_result = load_result(dump_file)
        else:
            metadata = inferred_dict['metadata']
            metadata.update({"start_date": datetime.datetime.now().astimezone().isoformat(timespec='seconds')})
            metadata['evaluate_result_path'] = dump_file
            evaluate_result = OrderedDict(
                metadata=metadata,
                evaluation=None,
                examples=[dict(index=i, input=e["input"], reference=e["reference"], prediction=inference_factory(e)) 
                        for i, e in enumerate(inferred_dict["examples"])]
            )
        start_time = time.time()
        human_evaluation_all = []
        for i, e in enumerate(evaluate_result["examples"]):
            logger.info(f"Evaluate the {i}-th example")
            human_evaluation = e.get("human_evaluation", None)
            auto_evaluation = e.get("auto_evaluation", None)
            if isinstance(human_evaluation, dict):
                human_evaluation = HumanEvaluationOutput.from_dict(human_evaluation)
            elif human_evaluation is None:
                human_evaluation = HumanEvaluationOutput()
            else:
                raise ValueError()

            if isinstance(auto_evaluation, dict):
                auto_evaluation = AutoEvaluationOutput.from_dict(auto_evaluation)
            elif auto_evaluation is None:
                auto_evaluation = AutoEvaluationOutput.from_dict(
                    automatic_metric_evaluate(
                        predictions=[e["prediction"]],
                        references = [e["reference"]],
                        input_texts = [e['input']],
                        target_styles = [transfer_mode2target_style(args.transfer_mode)],
                        task = args.task
                    )
                )
            else:
                raise ValueError()
            e["human_evaluation"] = human_evaluation.to_dict()
            e["auto_evaluation"] = auto_evaluation.to_dict()
            human_evaluation_all.append(human_evaluation)
        auto_evaluation_all = datasets.Dataset.from_list(evaluate_result["examples"]).to_dict()
        auto_evaluation_all = automatic_metric_evaluate(
            predictions= auto_evaluation_all["prediction"],
            references= auto_evaluation_all["reference"],
            target_styles=[transfer_mode2target_style(args.transfer_mode)]*len(evaluate_result["examples"]),
            input_texts=auto_evaluation_all['input'],
            task=args.task
        )
        evaluate_result["evaluation"] = OrderedDict(
            human_evaluation_all=HumanEvaluationOutput.sum(human_evaluation_all).to_dict(),
            auto_evaluation_all=auto_evaluation_all
        )
    except Exception as exception:
        raise exception
    finally:
        dump_result(evaluate_result, dump_file)
    time_cost = time.time() - start_time
    logger.info("--- %s seconds ---" % (round(time_cost)))
    logger.info(f"Finish evaluating and the results are dumped at '{dump_file}', time cost:{time_cost}!")


def automatic_metric_evaluate(predictions, references, input_texts, target_styles, task):
    evaluator = EvaluatorAPI(host= args.e_host, port=args.e_port)
    automatic_metric = evaluator.autometric(predictions=predictions,references=references,inputs=input_texts, target_styles=target_styles)
    logger.debug(automatic_metric)
    return automatic_metric


def reduction_synthesis(dataset, reduction_prompt, synthesis_prompt):
    dump_file = os.path.join(args.output_dir, "reduction_synthesis.txt")  
    start_time = int(time.time())
    success_count = 0
    try:
        if os.path.isfile(dump_file):
            result = load_result(dump_file)
        else:
            result = init_result(reduction_prompt) 
            result["prompt_format"] = {
                "reduction": result["prompt_format"],
                "synthesis": synthesis_prompt.prompt_format,
            }
            result["examples"] = dataset.to_list()
            for e in result["examples"]:
                e["reduction"] = None
                e["synthesis"] = None
        for i, e in enumerate(result["examples"]):
            input_text = e["input"]
            if e["reduction"] is None:
                logger.info(f"Reducte the {i}-th Input: {input_text}")
                e["reduction"] = self_refine(input_text, reduction_prompt, psc, target_style=Sentiment.NEUTRAL)
                e["reduction"]["evaluation"] = automatic_metric_evaluate(
                        predictions=[e["reduction"]["output"]],
                        references = [e["reference"]],
                        input_texts = [e['input']],
                        target_styles = [Sentiment.NEUTRAL],
                        task = args.task
                    )
            if e["synthesis"] is None:
                logger.info(f"Synthese the {i}-th Input: {input_text}")
                e["synthesis"] = self_refine(input_text, synthesis_prompt, psc, reduction_text=e["reduction"]['output'], target_style=args.transfer_mode.target)
            success_count += 1
    except Exception as exception:
        raise exception
    finally:
        time_cost = int(time.time()) - start_time
        metadata: dict = result["metadata"]
        metadata.update(vars(args))
        metadata.update({
            "total_example_num": len(result["examples"]),
            "last_time_cost": time_cost,
            "operation_times": metadata.get("operation_times", 0) + 1,
            "time_cost": metadata.get("time_cost", 0) + time_cost,
            "successful_num": success_count,
            "result_path": os.path.abspath(dump_file),
        })
        dump_result(result, dump_file)

    logger.debug(f"Finish reduction and synthesis: {time_cost}")
    return result

def rs_online(input_text):
    # input_text = "it isn't terrible, but it isn't very good either."
    prompts = get_prompts(args.approach, args.transfer_mode, args.task)
    reduction_result = self_refine(input_text, prompts[0], psc, target_style=Sentiment.NEUTRAL)

    result = self_refine(input_text, prompts[1], psc, reduction_text=reduction_result['output'],  target_style=args.transfer_mode.target)

    logger.debug(f"Reduction:{reduction_result['output']}")
    logger.debug(f"Output:{result['output']}")
    return reduction_result, result


def self_refine_tst(dataset, prompt):
    dump_file = os.path.join(args.output_dir, "self_refine.txt")
    start_time = time.time()
    try:
        if os.path.isfile(dump_file):
            result = load_result(dump_file)
        else:
            result = init_result(prompt)
            result["examples"] = dataset.to_list()
            for e in result["examples"]:
                e["self_refine"] = None
        success_count = 0
        for i, e in enumerate(result["examples"]):
            if e["self_refine"] is None:
                input_text = e["input"]
                logger.info(f"Inferring the {i}-th Input: {input_text}")
                self_refine_result = self_refine(input_text, prompt, psc, target_style=args.transfer_mode.target)
                # self_refine_result["input"] = input_text
                logger.info(f"Reduction Output: {self_refine_result['output']}")
                logger.info(f"Stop Event: {self_refine_result['stop_event']}")
                e["self_refine"] = self_refine_result
            success_count += 1 
    except Exception as exception:
        raise exception
    finally:
        time_cost = time.time() - start_time
        metadata = result["metadata"]
        metadata.update(vars(args))
        metadata.update({
            "total_example_num": len(result["examples"]),
            "last_time_cost": time_cost,
            "operation_times": metadata.get("operation_times", 0) + 1,
            "time_cost": metadata.get("time_cost", 0) + time_cost,
            "successful_num": success_count,
            "self_refine_tst_result_path": os.path.abspath(dump_file)
        })
        dump_result(result, dump_file)

    logger.info(f"Finish baseline (self_refine), time cost:{time_cost}!")
    return result


if __name__ == "__main__":
    hostname = socket.gethostname()
    share_hostname = "node_share"
    parser = argparse.ArgumentParser()
    # Parser arguments
    parser.add_argument('--host', type=str, help="The address of host supplying api service of inferring")
    parser.add_argument('--port', type=int, help="The port of host supplying api service of inferring")
    parser.add_argument('--e_host', type=str, help="The address of host supplying api service of evaluator")
    parser.add_argument('--e_port', type=int, help="The port of host supplying api service of evaluator")
    parser.add_argument('--model', type=str,
                        help="The name of model which is used.")
    parser.add_argument('--task', type=str, 
                        help="The name of task, it will be extract from the value of '--dataset_dir'.")
    parser.add_argument('--approach', type=str,
                        help="The name of approach", choices=list(Approach) )
    parser.add_argument('--transfer_mode', type=str,
                        help="The mode of sentiment transfer", choices=list(TransferMode))
    parser.add_argument('--dataset_dir', type=str,
                        help="The path to the dataset operated.")
    parser.add_argument('--temperature', type=float, default=0.6,
                        help="The temperature value for controlling randomness in generation. Defaults to 0.6.")
    parser.add_argument('--top_p', type=float, default=0.9,
                        help="The top-p sampling parameter for controlling diversity in generation.")
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help="The maximum sequence length for input prompts. Defaults to 1024.")
    parser.add_argument('--max_gen_len', type=int, default=96,
                        help="The maximum length of generated sequences. Defaults to 96.")
    parser.add_argument('--max_batch_size', type=int, default=4,
                        help="The maximum batch size for generating sequences. Defaults to 4.")
    parser.add_argument('--online_test', default=False, action='store_true',
                        help="Whether conduct online test, or process dataset if false")
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=str, default=1234)

    args = parser.parse_args()

    # Transform the name of args.approach
    args.approach = Approach(args.approach)

    # Transform the type of args.transfer_mode
    args.transfer_mode = TransferMode(args.transfer_mode)

    # Retrival the name of LLM and dataset
    search_dataset = re.search(f"{'|'.join(f'({t})' for t in list(Task))}", args.dataset_dir)

    generator: OllamaAPI = OllamaAPI(args.host, args.port)
    
    if args.model not in generator.models:
        # raise ValueError(f"The name ({args.model}) is inavilable at {generator.name}!")
        generator.pull_model(args.model)

    # Format the output_dir
    if isinstance(args.output_dir, str) and len(args.output_dir.strip()) > 0:
        args.output_dir = os.path.join("output" + args.output_dir)
    else:
        args.output_dir = "output"

    search_dataset = re.search(f"{'|'.join(f'({t})' for t in list(Task))}", args.dataset_dir)
    if search_dataset:
        args.task = Task(args.dataset_dir[search_dataset.start(): search_dataset.end()])
    else:
        raise ValueError(f"Can not retrival the name of dataset from the dataset_dir '{args.dataset_dir}'!")
    
    args.log_dir = args.output_dir
    args.output_dir = os.path.join(args.output_dir, share_hostname, f"{args.task}_{args.model.replace(":", "_")}_{args.transfer_mode}")

    # init logger
    logger = get_logger(name="main", dump_path=os.path.join(args.log_dir, hostname, f"{args.task}_{args.model.replace(":", "_")}_{args.transfer_mode}.log"))
    logger.info("**********environment**********")
    for k, v in os.environ.items():
        logger.info(f"{k}:\t{v}")
    logger.info("*****************************")

    logger.info("**********arguments**********")
    for k, v in vars(args).items():
        logger.info(f"{k}:\t{v}")
    logger.info("*****************************")

    assert isinstance(args.transfer_mode, TransferMode)
    logger.info(f"Load dataset of task ({args.task}) from '{args.dataset_dir}'")
    dataset = load_dataset(args.dataset_dir)[args.transfer_mode]
    try:
        tst()
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e

