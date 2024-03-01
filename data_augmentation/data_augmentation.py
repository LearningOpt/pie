
import html2text
import os
from transformers import GPT2Tokenizer
from tqdm import tqdm
import re
import contextlib
import joblib
import logging
import tempfile
import subprocess
import json
from typing import List, Union, Dict, Tuple, Any
import pandas as pd
import glob
import numpy as np
from collections import defaultdict
import random
import openai
import signal
import traceback
from tenacity import *
from transformers.utils import logging as transformers_logging
import copy
import pickle
import uuid
import traceback
import multiprocessing
import time


PATH_TO_KEY=""
PATH_TO_PIE=""

from multiprocessing import Manager
manager = Manager()
global rate_limiter
rate_limiter = manager.Queue()
rate_limiter.put((time.time(), 0))
from threading import Lock
global lock 
lock = Lock()

def setup_signal_handler():
    def handler(signum, frame):
        print(f"Process {multiprocessing.current_process().name} received SIGINT. Stack trace:")
        traceback.print_stack(frame)
        exit(0)
    signal.signal(signal.SIGINT, handler)

transformers_logging.set_verbosity(40)

with open(PATH_TO_KEY, "r") as fh:
    openai_key = fh.read().strip()
openai.api_key = openai_key

logging.basicConfig(level=logging.INFO)

global gpt_tokenizer
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

## TODO - use the annotated outputs from GEM5 and use that to sort by the fastest submissions

PATH_TO_HTML_PROMPTS=f"{PATH_TO_PIE}/data/Project_CodeNet/problem_descriptions/*.html"
PATH_TO_ALL_EXAMPLES=f"{PATH_TO_PIE}/data/codenet/metadata_dict_gem5_updated.jsonl"

global h
h = html2text.HTML2Text()
h.ignore_links = True
h.ignore_images = True
h.ignore_emphasis = True
h.ignore_tables = True
h.ignore_anchors = True

from bs4 import BeautifulSoup

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()
    return clean_text


CODE_ONLY_TEMPLATE: str = """
Description 1: ...
Code 1: ...
Description 2: ...
Code 2: ...
Now, can you generate a program that takes that same input as Code 2 in Code 3 but produces different outputs? Write it to be as novel as possible. 
Code 3: """


CODE_DESCRIPTION_PAIR_START_TEMPLATE: str = """
Description 1: ...
Code 1: ...
Description 2: ...
Code 2: ...
Now, can you generate a description for a new problem. It must take input from Description 2/Code 2 but will produce a different output than Code 2. Write it to be as different as novel as possible. Do not solve it yet.
Description 3:"""


CODE_DESCRIPTION_PAIR_END_TEMPLATE: str = """
Now, can you generate a program that solves Description 3? Write it to be as novel as possible and remember it needs to produce a different output than Code 2. 
Code 3:"""

class TimeoutException(Exception):
    pass

def alarm_handler(signum, frame):
    raise TimeoutException("Timed out!")

def canonicalize_newlines(text):
    return re.sub(r"[\n]+", "\n", text)

def canonicalize_newlines_2(text):
    # replaces 2 or more with 2 
    return re.sub(r"[\n]{2,}", "\n\n", text)

def paste_in_code(code, index: int, template: str) -> str:
    assert index in [1, 2]
    # new_template = re.sub(rf"Code {index}: ...", f"Code {index}: {code}\n\n", template)
    code = canonicalize_newlines(code)
    new_template = template.replace(f"Code {index}: ...", f"Code {index}: {code}\n\n")
    return new_template

def paste_in_description(description, index: int, template: str) -> str:
    assert index in [1, 2, 3]
    # new_template = re.sub(rf"Description {index}: ...", f"Description {index}: {description}\n\n", template)
    description = remove_html_tags(description)
    description = canonicalize_newlines_2(description)
    new_template = template.replace(f"Description {index}: ...", f"Description {index}: {description}\n\n")
    return new_template

def append_to_template(template: str, text: str) -> str:
    return template + text + "\n\n"


@retry(
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(openai.error.APIConnectionError),
)
def _get_responses(content, max_tokens=3000, top_p=1, n=1, temperature=0.0, model="gpt-4-0613"):
    
    message=[{"role": "user", "content": content}]
    
    try: 
        # Rate limiting logic
        
        tokens_needed = max_tokens * n
    
        global rate_limiter
        global lock

        while True:
            with lock:  # Automatically acquires and releases the lock
                if not rate_limiter.empty():
                    last_time, last_tokens = rate_limiter.get()
                else:
                    last_time, last_tokens = time.time(), 0

                current_time = time.time()

                if (current_time - last_time < 65) and (last_tokens + tokens_needed > 180000):
                    time_to_sleep = 65 - (current_time - last_time)
                    logging.info(f"Rate limit exceeded. Sleeping for {time_to_sleep} seconds, last_tokens: {last_tokens}, tokens_needed: {tokens_needed}")
                    
                    # Put the tokens back since we're going to sleep
                    rate_limiter.put((last_time, last_tokens))
                else:
                    # Update the queue and exit the loop
                    if current_time - last_time > 65:
                        rate_limiter.put((current_time, tokens_needed))
                        logging.info(f"Rate limit not exceeded, queue >60 seconds passed, it now now has {tokens_needed} tokens")
                    else: 
                        logging.info(f"Rate limit not exceeded, queue was <60 seconds and now has {last_tokens + tokens_needed} tokens")
                        rate_limiter.put((current_time, last_tokens + tokens_needed))
                    break

            if time_to_sleep > 0:
                time.sleep(time_to_sleep)


        response = openai.ChatCompletion.create(
        model=model,
        messages=message,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
        request_timeout=120,
        )
        choices = response.choices
        completions = [choice["message"]["content"] for choice in choices]
    except Exception as e:
        logging.warning(f"Failed to get response from openai with error {e}")
        stack_trace = traceback.format_exc()
        logging.warning(f"Stack trace: {stack_trace}")
        raise e
    return completions

def get_responses(content, max_tokens=3000, top_p=1, n=1, temperature=0.0, model="gpt-4-0613"):
    try: 
        completions = _get_responses(content, max_tokens=max_tokens, top_p=top_p, n=n, temperature=temperature, model=model)    
        print(f"collected responses successfully", flush=True)
    except Exception as e:
        logging.warning(f"Failed to get response from openai with error {e}")
        stack_trace = traceback.format_exc()
        logging.warning(f"Stack trace: {stack_trace}")
        completions = ["Openai timed out"]
    return completions


def test_responses():
    jokes = get_responses("tell me a joke", max_tokens=200, top_p=0.8, n=2, temperature=0.2)
    for j in jokes:
        print(f"Joke: {j}")
        print("\n")
        

PATH_TO_HTML_PROMPTS = f"{PATH_TO_PIE}/data/Project_CodeNet/problem_descriptions/*.html"

def get_all_html_promopts(prompt_pattern=PATH_TO_HTML_PROMPTS):
    html_prompt_paths = glob.glob(prompt_pattern)
    html_prompts = []
    problem_id_to_html = {}
    problem_id_to_text_description = {}
    html_prompts_text = []
    for path in html_prompt_paths:
        with open(path, "r") as fh:
            html_prompt = fh.read().strip()
        problem_id = os.path.basename(path).split(".")[0]
        html_prompts.append(html_prompt)
        problem_id_to_html[problem_id] = html_prompt

    return html_prompts, problem_id_to_html


def html_to_text(html_text):
    parsed = h.handle(html_text)
    if len(parsed) < 25: 
        return html_text
    return parsed


def get_gpt_length(text):
    return len(gpt_tokenizer.encode(text))


def get_all_text_prompts(prompt_pattern=PATH_TO_HTML_PROMPTS):
    n_too_short = 0
    problem_id_to_text_description = {}
    html_prompts_text = []
    html_prompts, problem_id_to_html = get_all_html_promopts(prompt_pattern=prompt_pattern)
    for problem_id, html_prompt in problem_id_to_html.items():
        text_prompt = html_to_text(html_prompt)
        problem_id_to_text_description[problem_id] = text_prompt
        if get_gpt_length(text_prompt) < 25: 
            # print(f"{'*' * 20} {pid} {'*' * 20}")
            # logging.warning(f"Problem {problem_id} has a short prompt: {text_prompt}")
            logging.warning(f"{'*' * 20} {problem_id} is too short {'*' * 20}\n")
            logging.warning(f"{text_prompt}\n")
            n_too_short += 1
        html_prompts_text.append(text_prompt)
    logging.warning(f"Found {n_too_short} prompts that are too short")
    return html_prompts_text, problem_id_to_text_description


def read_in_process_all_accepted_examples(path_to_examples=PATH_TO_ALL_EXAMPLES): 
    examples = pd.read_csv(path_to_examples)
    accepted = examples[examples["status"] == "Accepted"]
    accepted = accepted[~accepted["problem_id"].isin(missing)]
    accepted = accepted[accepted["problem_id"].isin(problem_id_to_text_description.keys())]
    accepted["text_description"] = accepted["problem_id"].apply(lambda x: problem_id_to_text_description[x])
    return accepted


TESTCASE_ROOT_DIR=f"{PATH_TO_PIE}/data/codenet/merged_test_cases/"


def compile_program(prog, problem_id, directory, prog_name="ref", verbose=False, gcc_opt_flag="-O2", timeout=45):
    code_path = os.path.join(directory, f"{problem_id}_{prog_name}.cpp")
    with open(code_path, "w") as f:
        f.write(prog)
    binary_path = os.path.join(directory, f"{problem_id}_{prog_name}.out")
    try: 
        p = subprocess.run(["g++", "-std=c++17", gcc_opt_flag, code_path, "-o", binary_path], capture_output=True, timeout=timeout)
        if p.returncode != 0:
            if verbose:
                logging.warning("gen failed to compile")
                logging.warning(p.stderr.decode("utf-8"))
            return False, binary_path, code_path
        return True, binary_path, code_path
    except subprocess.TimeoutExpired:
        if verbose:
            logging.warning("gen timed out")
        return False, binary_path, code_path
    

def get_output(binary_path, testcase_path, timeout=45, verbose=False):
    try: 
        p = subprocess.run([binary_path], stdin=open(testcase_path, "r"), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if verbose: 
            logging.warning("gen timed out")
        return "timeout"
    if p.returncode != 0:
        if verbose:
            logging.warning("gen failed to run")
        return "error"
    try: 
        gen_output = p.stdout.decode("utf-8").strip()
        return gen_output
    except UnicodeDecodeError:
        MAX_LENGTH = 1000
        if verbose:
            logging.warning(f"Failed to decode output for binary_path {binary_path} and testcase_path {testcase_path} output is {p.stdout[:MAX_LENGTH]}")

        return "error"
    
def check_program_is_correct(prog, problem_id, working_dir, testcase_root_dir=TESTCASE_ROOT_DIR, gcc_opt_flag="-O2", verbose=False, timeout=45):
    """
    returns (is_correct, binary_path, code_path)
    1. Compile and save to working_dir with name {problem_id}.cpp
    2. For each testcase, input is determined by the problem_id/input.{testcase_id}.txt, and output is determined by problem_id/output.{testcase_id}.txt
    3. If the output is not the same as the expected output, return False
    """
    is_compiled, binary_path, code_path = compile_program(prog, problem_id, working_dir, gcc_opt_flag=gcc_opt_flag, verbose=verbose, timeout=timeout)
    if not is_compiled:
        # print(f"Failed to compile program for problem_id {problem_id}")
        return False, binary_path, code_path
    testcases = glob.glob(os.path.join(testcase_root_dir, f"{problem_id}/input.*.txt"))
    testcase_ids = [int(t.split(".")[-2]) for t in testcases]
    for testcase_id, testcase_path in zip(testcase_ids, testcases):
        output = open(os.path.join(testcase_root_dir, f"{problem_id}/output.{testcase_id}.txt"), "r").read().strip()
        gen_output = get_output(binary_path, testcase_path, verbose=verbose, timeout=timeout)
        if gen_output == "timeout" or gen_output == "error":
            # print(f"Failed to run program for problem_id {problem_id} with a problem: {gen_output}")
            return False, binary_path, code_path
        if output != gen_output:
            # print(f"Failed to get correct output for problem_id {problem_id}, output is {gen_output} but expected {output}")
            return False, binary_path, code_path
    return True, binary_path, code_path

def check_program_runs_all_testcases(prog, problem_id, working_dir, prog_name="ref", testcase_root_dir=TESTCASE_ROOT_DIR, gcc_opt_flag="-O2", verbose=False, timeout=10):
    """
    returns (runs_all_testcases, binary_path, code_path)
    1. Compile and save to working_dir with name {problem_id}.cpp
    2. For each testcase, input is determined by the problem_id/input.{testcase_id}.txt
    3. If the program fails to run on any testcase, return False (ie p.returncode != 0)
    """
    # TODO: I think refactor to use bin_path??? 
    testcases = glob.glob(os.path.join(testcase_root_dir, f"{problem_id}/input.*.txt"))
    testcase_ids = [int(t.split(".")[-2]) for t in testcases]
    testcase_id_2_output = {}
    
    is_compiled, binary_path, code_path = compile_program(prog, problem_id, working_dir, prog_name=prog_name, gcc_opt_flag=gcc_opt_flag, verbose=verbose, timeout=timeout)
    try:     
        if not is_compiled:
            return False, binary_path, code_path, testcase_id_2_output
        for testcase_id, testcase_path in zip(testcase_ids, testcases):
            gen_output = get_output(binary_path, testcase_path, verbose=verbose, timeout=timeout)
            if gen_output == "timeout" or gen_output == "error":
                return False, binary_path, code_path, testcase_id_2_output
            testcase_id_2_output[testcase_id] = gen_output
        return True, binary_path, code_path, testcase_id_2_output
    except TimeoutException:
        return False, binary_path, code_path, testcase_id_2_output
    

def get_fastest_correct_examples_from_path(df_path, working_dir, timeout=45):
    return get_fastest_correct_examples(pd.read_json(df_path, lines=True, orient="records"), working_dir, timeout=timeout)

def get_fastest_correct_examples(df, working_dir, timeout=45):
    # groupby problem_id
    # sort by agg_runtime
    # drop all problem_ids where agg_runtime is nan or inf 
    # then for each problem id, greedily take the first one that is correct
    # return the list of problem ids
    _, problem_id_to_text_description = get_all_text_prompts()
    
    df = df[~df["agg_runtime"].isna()]
    df = df[~df["agg_runtime"].isin([np.inf, -np.inf])]
    df = df.groupby("problem_id").apply(lambda group: group.sort_values("agg_runtime"))
    problem_ids = list(set(df["problem_id"].values))
    problem_ids_2_fastest = {}
    for problem_id in tqdm(problem_ids, desc="get_fastest_correct_examples"):
        print(f"column names are {df.columns}")
        problem_df = df[df["problem_id"] == problem_id]
        problem_df = problem_df.sort_values("agg_runtime")
        for _, row in problem_df.iterrows():
            is_correct, binary_path, code_path = check_program_is_correct(
                row["code"], problem_id, working_dir, timeout=timeout, verbose=True
                )
            if is_correct:
                problem_ids_2_fastest[problem_id] = {
                    "code": row["code"],
                    "binary_path": binary_path,
                    "code_path": code_path,
                    "agg_runtime": row["agg_runtime"],
                    "num_submissions": len(problem_df), 
                    "n_tests": row["n_tests"],
                }
                text_description = problem_id_to_text_description.get(problem_id, None)
                problem_ids_2_fastest[problem_id]["text_description"] = text_description
                problem_ids_2_fastest[problem_id]["code_length"] = get_gpt_length(row["code"])
                problem_ids_2_fastest[problem_id]["text_description_length"] = get_gpt_length(text_description) if text_description is not None else 0
                problem_ids_2_fastest[problem_id]["total_length"] = problem_ids_2_fastest[problem_id]["code_length"] + problem_ids_2_fastest[problem_id]["text_description_length"] 
                break
    return problem_ids_2_fastest

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
        

def _process_problem_id(problem_id, df, problem_id_to_text_description, working_dir, timeout=45, max_submissions=10):
    problem_df = df[df["problem_id"] == problem_id]
    del df
    problem_df = problem_df.sort_values("agg_runtime").iloc[:max_submissions]

    for _, row in problem_df.iterrows():
        is_correct, binary_path, code_path = check_program_is_correct(
            row["code"], problem_id, working_dir, timeout=timeout
        )
        if is_correct:
            text_description = problem_id_to_text_description.get(problem_id, None)
            return problem_id, {
                "code": row["code"],
                "binary_path": binary_path,
                "code_path": code_path,
                "agg_runtime": row["agg_runtime"],
                "num_submissions": len(problem_df),
                "text_description": text_description,
                "code_length": get_gpt_length(row["code"]),
                "text_description_length": get_gpt_length(text_description) if text_description is not None else 0,
                "total_length": get_gpt_length(row["code"]) + get_gpt_length(text_description) if text_description is not None else 0, 
                "n_tests": row["n_tests"],
            }
    del problem_df
    return None


def get_fastest_correct_examples_parallel_from_path(df_path, working_dir, timeout=45):
    return get_fastest_correct_examples_parallel(pd.read_json(df_path, lines=True, orient="records"), working_dir, timeout=timeout)


def get_fastest_correct_examples_parallel(df, working_dir, timeout=45):
    _, problem_id_to_text_description = get_all_text_prompts()
    
    df = df[~df["agg_runtime"].isna()]
    df = df[~df["agg_runtime"].isin([np.inf, -np.inf])]
    df = df[df["status"] == "Accepted"]
    df = df.groupby("problem_id").apply(lambda group: group.sort_values("agg_runtime"))

    problem_ids = list(set(df["problem_id"].values))
    
    df_chunks = [df[df["problem_id"] == problem_id] for problem_id in problem_ids]
    del df

    results = []
    with tqdm_joblib(tqdm(desc="Processing problem_ids", total=len(problem_ids))) as progress_bar:
        results = joblib.Parallel(n_jobs=20)(
            joblib.delayed(_process_problem_id)(problem_id, df, problem_id_to_text_description, working_dir, timeout=timeout)
            for problem_id, df in zip(problem_ids, df_chunks)
        )
        # for result in joblib.Parallel(n_jobs=20)(
        #     joblib.delayed(_process_problem_id)(problem_id, df, problem_id_to_text_description, working_dir, timeout=timeout)
        #     for problem_id, df in zip(problem_ids, df_chunks)
        # ):
        #     # progress_bar.update(1)
        #     import pdb;
        #     pdb.set_trace()
        #     if result is not None:
        #         results.append(result)
            

    results = [result for result in results if result is not None]
    problem_ids_2_fastest = {problem_id: result for problem_id, result in results}
    
    return problem_ids_2_fastest


def build_prompt(problem_ids_2_fastest, prompt_type, candidate_problem_ids: List[str], candidate_problem_id: Union[str, None] = None, max_prompt_length=10000):
    assert prompt_type in ["code_only", "code_description_pair"], f"prompt_type must be one of ['code_only', 'code_description_pair'], but got {prompt_type}"
    template = CODE_ONLY_TEMPLATE if prompt_type == "code_only" else CODE_DESCRIPTION_PAIR_START_TEMPLATE
    
    if candidate_problem_id is None:
        candidate_problem_id = random.choice(candidate_problem_ids)
    candidate_problem_dict = problem_ids_2_fastest[candidate_problem_id]
    code_2 = candidate_problem_dict["code"]
    description_2 = candidate_problem_dict["text_description"]
    total_length = candidate_problem_dict["total_length"]
    template = paste_in_code(code_2, 2, template)
    template = paste_in_description(description_2, 2, template)
    
    problem_id_1 = random.choice(list(problem_ids_2_fastest.keys()))
    problem_id_1_length = problem_ids_2_fastest[problem_id_1]["total_length"]
    problem_id_1_description_length = problem_ids_2_fastest[problem_id_1]["text_description_length"]
    max_tries = 25
    while(problem_id_1_length + total_length > max_prompt_length or problem_id_1_description_length == 0):
        problem_id_1 = random.choice(list(problem_ids_2_fastest.keys()))
        problem_id_1_length = problem_ids_2_fastest[problem_id_1]["total_length"]
        problem_id_1_description_length = problem_ids_2_fastest[problem_id_1]["text_description_length"]
        max_tries -= 1
        print(f"max_tries is {max_tries} for problem_id_1 {problem_id_1}")
        if max_tries == 0:
            logging.warning(f"Failed to find a problem_id_1 that is short enough to fit in the prompt with problem_id_1_length {problem_id_1_length} and total_length {total_length}")
            return None, None, None, None
    code_1 = problem_ids_2_fastest[problem_id_1]["code"]
    description_1 = problem_ids_2_fastest[problem_id_1]["text_description"]
    template = paste_in_code(code_1, 1, template)
    template = paste_in_description(description_1, 1, template)
    gpt_length = get_gpt_length(template)
    return template, candidate_problem_id, problem_id_1, gpt_length


def test_build_prompt(problem_ids_2_fastest, candidate_problem_ids: List[str], max_prompt_length=5000):
    template, candidate_problem_id, problem_id_1, gpt_length = build_prompt(problem_ids_2_fastest, prompt_type="code_only", candidate_problem_ids=candidate_problem_ids, max_prompt_length=max_prompt_length)
    assert gpt_length <= max_prompt_length
    logging.info(f"Built prompt for code_only with length {gpt_length}\n\nproblem id 1: {problem_id_1}\n\ncandidate problem id (ie problem id 2): {candidate_problem_id}\n\nand the prompt is:\n\n{template}")
    template, candidate_problem_id, problem_id_1, gpt_length = build_prompt(problem_ids_2_fastest, prompt_type="code_description_pair", candidate_problem_ids=candidate_problem_ids, max_prompt_length=max_prompt_length)
    assert gpt_length <= max_prompt_length
    logging.info(f"Built prompt for code_description_pair with length {gpt_length}\n\nproblem id 1: {problem_id_1}\n\ncandidate problem id (ie problem id 2): {candidate_problem_id}\n\nand the prompt is:\n\n{template}")
    return True


def generate_code_descriptipon_pair(problem_ids_2_fastest, candidate_problem_ids: List[str], candidate_problem_id: Union[str, None] = None, max_prompt_length=5000, max_tokens=1600, top_p=1, temperature=0.6, n=1, model="gpt-3.5-turbo-16k-0613"):
    template, candidate_problem_id, problem_id_1, gpt_length = build_prompt(problem_ids_2_fastest, prompt_type="code_description_pair", candidate_problem_ids=candidate_problem_ids, candidate_problem_id=candidate_problem_id, max_prompt_length=max_prompt_length)
    if template is None:
        return None, None, None, None
    description = get_responses(template, max_tokens=max_tokens - gpt_length - 10, top_p=top_p, n=1, temperature=temperature, model=model)[0]
    if description == "Openai timed out":
        print("Openai timed out for problem_id_1 {} and candidate_problem_id {}".format(problem_id_1, candidate_problem_id))
        return None, description, candidate_problem_id, problem_id_1, template
    template = append_to_template(template, description) 
    template = append_to_template(template, CODE_DESCRIPTION_PAIR_END_TEMPLATE)
    template_gpt_length = get_gpt_length(template)
    responses = get_responses(template, max_tokens=max_tokens - template_gpt_length - 10, top_p=top_p, n=n, temperature=temperature, model=model)
    for code in responses:
        if code == "Openai timed out":
            logging.warning("Openai timed out for problem_id_1 {} and candidate_problem_id {}".format(problem_id_1, candidate_problem_id))
    return responses, description, candidate_problem_id, problem_id_1, template
            


def test_generate_code_descriptipon_pair(problem_ids_2_fastest, candidate_problem_ids: List[str], max_prompt_length=5000, max_tokens=16000, top_p=1, temperature=0.6, model="gpt-3.5-turbo-16k-0613"):
    code, description, candidate_problem_id, problem_id_1, template = generate_code_descriptipon_pair(problem_ids_2_fastest, candidate_problem_ids, candidate_problem_id=None, max_prompt_length=max_prompt_length, max_tokens=max_tokens, top_p=top_p, temperature=temperature, model=model)
    logging.info(f"Generated code from gpt-3.5-turbo-16k-0613 with length {get_gpt_length(code)}\n\nproblem id 1: {problem_id_1}\n\ncandidate problem id (ie problem id 2): {candidate_problem_id}\n\nand the prompt is:\n\n{template}\n\nthe synthesized description is:\n\n{description}\n\nand the synthesized code is:\n\n{code}")
    return True


def generate_code_only(problem_ids_2_fastest, candidate_problem_ids: List[str], candidate_problem_id: Union[str, None] = None, max_prompt_length=10000, max_tokens=16000, top_p=1, temperature=0.6, n=1, model="gpt-3.5-turbo-16k-0613"):
    print(f"making prompt for candidate_problem_id {candidate_problem_id}", flush=True)
    template, candidate_problem_id, problem_id_1, gpt_length = build_prompt(problem_ids_2_fastest, prompt_type="code_only", candidate_problem_ids=candidate_problem_ids, candidate_problem_id=candidate_problem_id, max_prompt_length=max_prompt_length)
    print(f"made prompt for candidate_problem_id {candidate_problem_id} and problem_id_1 {problem_id_1} with length {gpt_length}", flush=True)
    if template is None:
        return None, None, None, None
    # import pdb; pdb.set_trace()
    responses = get_responses(template, max_tokens=max_tokens - gpt_length - 10, top_p=top_p, n=n, temperature=temperature, model=model)
    print(f"got responses for candidate_problem_id {candidate_problem_id} and problem_id_1 {problem_id_1}", flush=True)
    for code in responses:
        if code == "Openai timed out":
            logging.warning("Openai timed out for problem_id_1 {} and candidate_problem_id {}".format(problem_id_1, candidate_problem_id))
    return responses, candidate_problem_id, problem_id_1, template


def test_generate_code_only(problem_ids_2_fastest, candidate_problem_ids: List[str], max_prompt_length=5000, max_tokens=16000, top_p=1, temperature=0.6, model="gpt-3.5-turbo-16k-0613"):
    code, candidate_problem_id, problem_id_1, template = generate_code_only(problem_ids_2_fastest, candidate_problem_ids, candidate_problem_id=None, max_prompt_length=max_prompt_length, max_tokens=max_tokens, top_p=top_p, temperature=temperature, model=model)
    logging.info(f"Generated code from gpt-3.5-turbo-16k-0613 with length {get_gpt_length(code)}\n\nproblem id 1: {problem_id_1}\n\ncandidate problem id (ie problem id 2): {candidate_problem_id}\n\nand the prompt is:\n\n{template}\n\nand the synthesized code is:\n\n{code}")
    return True
        
        
def generate_novel_code(problem_ids_2_fastest,  problem_ids_2_all_outputs, working_dir, candidate_problem_ids: List[str], generation_strategy = "code_only", max_prompt_length=5000, max_tokens=16000, top_p=1, n=1, temperature=0.6, model="gpt-3.5-turbo-16k-0613"):

    assert generation_strategy in ["code_only", "code_description_pair"], f"generation_strategy must be one of ['code_only', 'code_description_pair'], but got {generation_strategy}"
    results = []
    for i, candidate_problem_id in tqdm(enumerate(candidate_problem_ids), desc="generate_novel_code"):
        _results = _inner_generate_novel_code(problem_ids_2_fastest, 
                                            problem_ids_2_all_outputs,
                                            working_dir, 
                                            candidate_problem_ids, 
                                            candidate_problem_id=candidate_problem_id, 
                                            prog_name=f"openai_gen_{i}",
                                            generation_strategy=generation_strategy, 
                                            max_prompt_length=max_prompt_length, 
                                            n=n,
                                            max_tokens=max_tokens, top_p=top_p, temperature=temperature, model=model)
        results.extend(_results)
        
    return results


def _inner_generate_novel_code(problem_ids_2_fastest, problem_ids_2_all_outputs, working_dir, candidate_problem_ids: List[str], candidate_problem_id: Union[str, None] = None, prog_name="openai_gen", generation_strategy = "code_only", max_prompt_length=5000, max_tokens=1600, top_p=1, temperature=0.6, n=1, model="gpt-3.5-turbo-16k-0613"):
    # import pdb; pdb.set_trace()
    # print(f"Generating novel code for candidate_problem_id {candidate_problem_id} with prog_name {prog_name}")
    non_picklable = check_picklable(locals())
    if non_picklable:
        print("Found non-picklable variables:", non_picklable, flush=True)
    else:
        print("Found no non-picklable variables in _inner_generate_novel_code", flush=True)

    if generation_strategy == "code_only":
        responses, candidate_problem_id, problem_id_1, template = generate_code_only(problem_ids_2_fastest, candidate_problem_ids, candidate_problem_id=candidate_problem_id, max_prompt_length=max_prompt_length, max_tokens=max_tokens, top_p=top_p, n=n, temperature=temperature, model=model)
        description = "not generated"
    else:
        responses, description, candidate_problem_id, problem_id_1, template = generate_code_descriptipon_pair(problem_ids_2_fastest, candidate_problem_ids, candidate_problem_id=candidate_problem_id, max_prompt_length=max_prompt_length, max_tokens=max_tokens, top_p=top_p, temperature=temperature, n=n, model=model)
    # if code is not None:
    #     is_different, binary_path, code_path, matching_problem_ids = test_output_against_all(candidate_problem_id, code, problem_ids_2_all_outputs, working_dir, prog_name=prog_name, testcase_root_dir=TESTCASE_ROOT_DIR, gcc_opt_flag="-O2", verbose=False, timeout=10)
    # else: 
    #     is_different = False
    #     binary_path = None
    #     code_path = None
    #     matching_problem_ids = []
    # result = {
    #     "candidate_problem_id": candidate_problem_id,
    #     "code": code,
    #     "description": description,
    #     "is_different": is_different,
    #     "binary_path": binary_path,
    #     "code_path": code_path,
    #     "matching_problem_ids": list(matching_problem_ids),
    #     "template": template
    # }
    # return result
    
    ## with multiple_responses; responses := [code_1, code_2, ...]
    n_successful = len([response for response in responses if response != "Openai timed out" and response is not None])
    template_cost, template_len = calculate_cost_of_template(template)
    template_cost_per = template_cost / n_successful if n_successful > 0 else 0
    results = [] 
    for i, code in enumerate(responses):
        _prog_name = f"{prog_name}_{i}_{uuid.uuid4()}"
        if code is not None and code != "Openai timed out":
            is_different, binary_path, code_path, matching_problem_ids = test_output_against_all(candidate_problem_id, code, problem_ids_2_all_outputs, working_dir, prog_name=_prog_name, testcase_root_dir=TESTCASE_ROOT_DIR, gcc_opt_flag="-O2", verbose=False, timeout=10)
            generation_cost, generation_length = calculate_cost_of_generation(code)
            total_cost = generation_cost + template_cost_per
        else: 
            is_different = False
            binary_path = None
            code_path = None
            matching_problem_ids = []
            generation_cost = 0
            total_cost = 0
            generation_length = 0
        result = {
            "candidate_problem_id": candidate_problem_id,
            "code": code,
            "description": description,
            "is_different": is_different,
            "binary_path": binary_path,
            "code_path": code_path,
            "matching_problem_ids": list(matching_problem_ids),
            "template": template, 
            "generation_cost": generation_cost,
            "total_cost": total_cost, 
            "template_len": template_len,
            "generation_len": generation_length
        }
        results.append(result)
    return results
                
            
def calculate_cost_of_template(template, cost_per_1k=0.003):
    gpt_length = get_gpt_length(template)
    return gpt_length * cost_per_1k / 1000, gpt_length

def calculate_cost_of_generation(code, cost_per_1k=0.004):
    gpt_length = get_gpt_length(code)
    return gpt_length * cost_per_1k / 1000, gpt_length


def project_total_cost(n_api_calls, average_template_len = 6000, average_code_len = 2000, samples_per_call = 10, cost_per_1k_template=0.003, cost_per_1k_code=0.004):
    total_cost = n_api_calls * (average_template_len * cost_per_1k_template + average_code_len * cost_per_1k_code * samples_per_call) / 1000
    return total_cost



def parallel_generate_novel_code(problem_ids_2_fastest,  problem_ids_2_all_outputs, working_dir, candidate_problem_ids: List[str], generation_strategy = "code_only", max_prompt_length=5000, max_tokens=16000, top_p=1, temperature=0.6, n=1, model="gpt-3.5-turbo-16k-0613"):
    assert generation_strategy in ["code_only", "code_description_pair"], f"generation_strategy must be one of ['code_only', 'code_description_pair'], but got {generation_strategy}"
    results = []
    j = 0; 
    TOKENS_PER_MIN = 180000
    tokens_per_call = max_tokens * n
    n_api_calls = len(candidate_problem_ids) * tokens_per_call / TOKENS_PER_MIN
    
    with tqdm_joblib(tqdm(desc="Processing candidate_problem_ids", total=len(candidate_problem_ids))) as progress_bar:
        for _results in joblib.Parallel(n_jobs=-1, backend='threading')(joblib.delayed(_inner_generate_novel_code)(
            problem_ids_2_fastest, 
            problem_ids_2_all_outputs,
            working_dir, 
            candidate_problem_ids, 
            candidate_problem_id=candidate_problem_id, 
            prog_name=f"openai_gen_{i}",
            generation_strategy=generation_strategy, 
            max_prompt_length=max_prompt_length,
            max_tokens=max_tokens, 
            top_p=top_p,
            temperature=temperature,
            n=n,
            model=model
            ) for i, candidate_problem_id in enumerate(candidate_problem_ids)):
            j+=1
            # if (j % 5) == 0:
            #     logging.info(f"Processed {j} candidate_problem_ids")
            #     logging.info(f"Most recently generated code was {result['code']} for candidate_problem_id {result['candidate_problem_id']}")

            results.extend(_results)
            
    check_picklable(locals())
        
    return results


def select_n_candidates(problem_ids_2_fastest, n=10, take_top=True):
    for problem_id, problem_id_dict in problem_ids_2_fastest.items():
        problem_id_dict["n_tests"] = get_n_tests(problem_id)
    problem_ids_2_fastest = pd.DataFrame(problem_ids_2_fastest).T
    ## n_tests >= 20
    print(f"column names are {problem_ids_2_fastest.columns}")
    problem_ids_2_fastest = problem_ids_2_fastest[problem_ids_2_fastest["n_tests"] >= 20]
    if take_top:
        sample = problem_ids_2_fastest.sort_values("num_submissions", ascending=False).sample(n=n)
    else:
        # take a random sample
        sample = problem_ids_2_fastest.sample(n=n)
    return sample.to_dict("index")


def get_all_outputs_for_problem_id(problem_id, 
                                   problem_ids_2_fastest,
                                   testcase_root_dir=TESTCASE_ROOT_DIR, 
                                   gcc_opt_flag="-O2", verbose=False, position=0, timeout=45):
    testcases = glob.glob(f"/home/alex/Documents/PennPhD/learning2perf/data/codenet/merged_test_cases/{problem_id}/input*.txt")
    testcase_ids = [int(t.split(".")[-2]) for t in testcases]
    # sort testcases by testcase_id descending
    testcase_ids, testcases = zip(*sorted(zip(testcase_ids, testcases), key=lambda x: x[0], reverse=True))
    testcase_ids = list(testcase_ids)[:20]
    testcases = list(testcases)[:20]
    
    testcase_2_all_outputs = {}
    pbar = tqdm(total=len(testcase_ids) * len(problem_ids_2_fastest), position=position, desc=f"get_all_outputs_for_problem_id_{problem_id}")
    for testcase_id, testcase_path in zip(testcase_ids, testcases): 
        output_dict = defaultdict(list)
        for _problem_id, problem_id_dict in problem_ids_2_fastest.items(): 
            output = get_output(problem_id_dict["binary_path"], testcase_path, verbose=verbose, timeout=timeout)
            output_dict[output].append(_problem_id)
            pbar.update(1)
        testcase_2_all_outputs[testcase_id] = output_dict
    return problem_id, testcase_2_all_outputs

def _inner_get_output(_problem_id, problem_id_dict, testcase_path, verbose=False, timeout=45):
    output = get_output(problem_id_dict["binary_path"], testcase_path, verbose=verbose, timeout=timeout)
    return _problem_id, output

def parallel_get_all_outputs_for_problem_id(problem_id,
                                            problem_ids_2_fastest,
                                            testcase_root_dir=TESTCASE_ROOT_DIR,
                                            gcc_opt_flag="-O2", verbose=False, timeout=45):
    """
    Parallelize over the inner-loop of _problem_id, problem_id_dict in problem_ids_2_fastest.items()
    """
    testcases = glob.glob(f"/home/alex/Documents/PennPhD/learning2perf/data/codenet/merged_test_cases/{problem_id}/input*.txt")
    testcase_ids = [int(t.split(".")[-2]) for t in testcases]  
    testcase_ids, testcases = zip(*sorted(zip(testcase_ids, testcases), key=lambda x: x[0], reverse=True))
    testcase_ids = list(testcase_ids)[:20]
    testcases = list(testcases)[:20]
    
    testcase_2_all_outputs = {}
    pbar = tqdm(total=len(testcase_ids) * len(problem_ids_2_fastest), desc=f"get_all_outputs_for_problem_id_{problem_id}")
    with tqdm_joblib(tqdm(desc=f"get_all_outputs_for_problem_id_{problem_id}", total=len(testcase_ids) * len(problem_ids_2_fastest))) as progress_bar:
        for testcase_id, testcase_path in zip(testcase_ids, testcases): 
            output_dict = defaultdict(list)
            for _problem_id, output in joblib.Parallel(n_jobs=-1)(joblib.delayed(_inner_get_output)(
                _problem_id, 
                problem_id_dict, 
                testcase_path, 
                verbose=verbose, 
                timeout=timeout
                ) for _problem_id, problem_id_dict in problem_ids_2_fastest.items()):
                output_dict[output].append(_problem_id)
                # pbar.update(1)
            testcase_2_all_outputs[testcase_id] = output_dict
    return problem_id, testcase_2_all_outputs


def inner_parallel_get_all_outputs_for_problem_ids(problem_ids, 
                                                    problem_ids_2_fastest,
                                                    testcase_root_dir=TESTCASE_ROOT_DIR,
                                                    gcc_opt_flag="-O2", verbose=False, timeout=45):
    problem_ids_2_all_outputs = {}
    for problem_id in tqdm(problem_ids, desc="outer loop over problem_ids"):
        _, result = parallel_get_all_outputs_for_problem_id(problem_id, problem_ids_2_fastest, testcase_root_dir=testcase_root_dir, gcc_opt_flag=gcc_opt_flag, verbose=verbose, timeout=timeout)
        problem_ids_2_all_outputs[problem_id] = result
    return problem_ids_2_all_outputs
        
        
def get_all_outputs_for_problem_ids(problem_ids,
                                      problem_ids_2_fastest,
                                        testcase_root_dir=TESTCASE_ROOT_DIR,
                                        gcc_opt_flag="-O2", verbose=False, timeout=45):
    problem_ids_2_all_outputs = {}
    for problem_id in tqdm(problem_ids):
        _, result = get_all_outputs_for_problem_id(problem_id, problem_ids_2_fastest, testcase_root_dir=testcase_root_dir, gcc_opt_flag=gcc_opt_flag, verbose=verbose, timeout=timeout)
        problem_ids_2_all_outputs[problem_id] = result
    return problem_ids_2_all_outputs


def parallel_get_all_outputs_for_problem_ids(problem_ids,
                                        problem_ids_2_fastest,
                                        testcase_root_dir=TESTCASE_ROOT_DIR,
                                        gcc_opt_flag="-O2", verbose=False, timeout=45):
    problem_ids_2_all_outputs = {}
    for problem_id, result in joblib.Parallel(n_jobs=-1)(joblib.delayed(get_all_outputs_for_problem_id)(
        problem_id, 
        problem_ids_2_fastest, 
        testcase_root_dir=testcase_root_dir, 
        gcc_opt_flag=gcc_opt_flag, 
        verbose=verbose, 
        position=idx, 
        timeout=timeout
        ) for idx, problem_id in enumerate(problem_ids)):
        
        problem_ids_2_all_outputs[problem_id] = result
        
    return problem_ids_2_all_outputs


def get_n_tests(problem_id): 
    testcases = glob.glob(f"/home/alex/Documents/PennPhD/learning2perf/data/codenet/merged_test_cases/{problem_id}/input*.txt")
    return len(testcases)


def test_output_against_all(problem_id, 
                            code, 
                            problem_ids_2_all_outputs,
                            working_dir,
                            prog_name="openai_gen",
                            testcase_root_dir=TESTCASE_ROOT_DIR,
                            gcc_opt_flag="-O2", verbose=False, 
                            timeout=10):
    logging.info(f"Testing that program {prog_name} for problem_id {problem_id} compiles and runs on all testcases")
    # logging.info(f"problem_ids_2_all_outputs keys are {problem_ids_2_all_outputs.keys()}")
    # logging.info(f"problem_ids_2_all_outputs[problem_id] keys are {problem_ids_2_all_outputs[problem_id].keys()}")
    # print(f"problem_ids_2_all_outputs: {problem_ids_2_all_outputs}")
    problem_id_output_dict = problem_ids_2_all_outputs[problem_id]
    
    # problem_id_output_dict is a nested dict {test_case: {output: [problem_ids]}, } we want all problem_ids
    matching_problem_ids = set()
    # import pdb; pdb.set_trace()
    for testcase_id, output_dict in problem_id_output_dict.items():
        for output, problem_ids in output_dict.items():
            matching_problem_ids |= set(problem_ids)
    logging.info(f"there are {len(matching_problem_ids)} problem ids to test against")
    testcases = glob.glob(f"/home/alex/Documents/PennPhD/learning2perf/data/codenet/merged_test_cases/{problem_id}/input*.txt")
    testcase_ids = [int(t.split(".")[-2]) for t in testcases]
    # sort testcases by testcase_id descending
    testcase_ids, testcases = zip(*sorted(zip(testcase_ids, testcases), key=lambda x: x[0], reverse=True))
    testcase_ids = list(testcase_ids)[:20]
    testcases = list(testcases)[:20]
    # print(f"Testing that program {prog_name} for problem_id {problem_id} compiles and runs on all testcases")
    compiles_and_runs, binary_path, code_path, testcase_id_2_outputs = check_program_runs_all_testcases(code, problem_id, working_dir, prog_name=prog_name, testcase_root_dir=testcase_root_dir, gcc_opt_flag=gcc_opt_flag, verbose=verbose, timeout=timeout)
    if not compiles_and_runs:
        return False, binary_path, code_path, matching_problem_ids
    # print(f"Testing that program {prog_name} for problem_id {problem_id} produces unique outputs on all testcases")
    for testcase_id, testcase_path in zip(testcase_ids, testcases):
        
        gen_output = testcase_id_2_outputs[testcase_id]
        # if testcase_id == "100": 
        #     logging.info(f"Testing that program {prog_name} for problem_id {problem_id} produces unique outputs on all testcases with output {gen_output}")
        #     logging.info(f"problem_id_output_dict[testcase_id] is {problem_id_output_dict[testcase_id]}")
        if gen_output == "timeout" or gen_output == "error":
            raise ValueError(f"gen_output is {gen_output}, but should not be")
        if gen_output not in problem_id_output_dict[testcase_id]:
            return True, binary_path, code_path, set()
        equivalent_problem_ids = problem_id_output_dict[testcase_id][gen_output]
        neq_problem_ids = matching_problem_ids - set(equivalent_problem_ids)
        matching_problem_ids -= neq_problem_ids
        if matching_problem_ids == set():
            return True, binary_path, code_path, matching_problem_ids
    logging.info(f"problem_id {problem_id}/{prog_name} is not unique with matching_problem_ids {matching_problem_ids}")
    return False, binary_path, code_path, matching_problem_ids
    
from collections import defaultdict

def find_always_co_located(lst_of_lsts):
    # Step 1: Initialize a dictionary to store co-located strings
    co_located_dict = defaultdict(lambda: frozenset())

    # Step 2: Iterate through each sublist to populate the dictionary
    for sub_list in lst_of_lsts:
        sub_list_set = frozenset(sub_list)
        for elem in sub_list_set:
            if co_located_dict[elem]:
                co_located_dict[elem] &= sub_list_set # shrink the element's co-located elements
            else:
                co_located_dict[elem] = sub_list_set

    # Step 3: Filter unique always co-located sets
    unique_always_co_located = set()
    for key in co_located_dict.keys():
        if len(co_located_dict[key]) > 1:
            unique_always_co_located.add(co_located_dict[key])

    return list(unique_always_co_located)

def calculate_remaining_strings(total_unique_strings, co_located_sets):
    total_strings_in_co_located = set()
    for s in co_located_sets:
        total_strings_in_co_located |= s
    # import pdb; pdb.set_trace()
    return total_unique_strings - len(total_strings_in_co_located) + len(co_located_sets)

def calculate_output_agreement(problem_ids, 
                               problem_ids_2_fastest,
                               testcase_root_dir=TESTCASE_ROOT_DIR,
                               gcc_opt_flag="-O2", verbose=False, timeout=45):
    n_total_programs = len(problem_ids_2_fastest)
    # problem_ids_2_all_outputs = inner_parallel_get_all_outputs_for_problem_ids(problem_ids, problem_ids_2_fastest, testcase_root_dir=testcase_root_dir, gcc_opt_flag=gcc_opt_flag, verbose=verbose, timeout=timeout)
    # use parallel_get_all_outputs_for_problem_id
    problem_ids_2_all_outputs = parallel_get_all_outputs_for_problem_ids(problem_ids, problem_ids_2_fastest, testcase_root_dir=testcase_root_dir, gcc_opt_flag=gcc_opt_flag, verbose=verbose, timeout=timeout)
    ## flatten problem_ids_2_all_outputs into a list of lists
    all_outputs = []
    for problem_id, testcase_2_all_outputs in problem_ids_2_all_outputs.items():
        for testcase, outputs in testcase_2_all_outputs.items():
            for output, problem_ids in outputs.items():
                all_outputs.append(problem_ids)
    ## find all always co-located outputs
    # import pdb; pdb.set_trace()
    always_co_located = find_always_co_located(all_outputs)
    logging.info(f"Found {len(always_co_located)} always co-located outputs")
    logging.info(f"Always co-located outputs: {always_co_located}")
    ## calculate the number of unique outputs
    total_unique_programs = calculate_remaining_strings(n_total_programs, always_co_located)
    # print(f"Total number of unique programs: {total_unique_programs} from {n_total_programs} total programs which is {total_unique_programs / n_total_programs * 100:.2f}%")
    return total_unique_programs, n_total_programs, total_unique_programs / n_total_programs * 100, always_co_located

def check_picklable(scope_dict):
    non_picklable = {}
    
    for name, obj in scope_dict.items():
        try:
            pickle.dumps(obj)
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            non_picklable[name] = str(e)
            
    return non_picklable

def convert_str_tc_to_int_tc(problem_ids_2_all_outputs): 
    problem_ids_2_all_outputs_int = {}
    for problem_id, testcase_2_all_outputs in problem_ids_2_all_outputs.items():
        testcase_2_all_outputs_int = {}
        for testcase, outputs in testcase_2_all_outputs.items():
            testcase_2_all_outputs_int[int(testcase)] = outputs
        problem_ids_2_all_outputs_int[problem_id] = testcase_2_all_outputs_int
    return problem_ids_2_all_outputs_int

def main(args): 
    setup_signal_handler()
    non_picklable_locals = check_picklable(locals())
    print(f"Non-picklable local variables: {non_picklable_locals}")
    assert (args.total_iterations / 10) == int(args.total_iterations / 10), "total_iterations must be a multiple of 10"
    # working_dir = "/home/alex/Documents/PennPhD/learning2perf/data/data_augmentation/working_dir"
    
    projected_total_cost = project_total_cost(n_api_calls=args.total_iterations, average_template_len = 5000, average_code_len = 1000, samples_per_call = args.num_samples)
    logging.info(f"Projected total cost: {projected_total_cost}")
    
    experiment_name=f"openai_gen_{args.generation_strategy}_top_p_{args.top_p}_temperature_{args.temperature}_api_calls_{args.total_iterations}_samples_per_{args.num_samples}"
    working_dir = args.working_dir 
    results_dir_root = args.results_dir_root
    os.makedirs(results_dir_root, exist_ok=True)
    results_dir=results_dir_root + "/" + experiment_name
    os.makedirs(results_dir, exist_ok=True)
        
    
    os.makedirs(working_dir, exist_ok=True)
    if os.path.exists(os.path.join(working_dir, "fast_correct_examples.json")): 
        with open(os.path.join(working_dir, "fast_correct_examples.json"), "r") as f:
            fast_correct_examples = json.load(f)
    else: 
        logging.info("Generating fast_correct_examples")
        fast_correct_examples = get_fastest_correct_examples_parallel_from_path(
            PATH_TO_ALL_EXAMPLES, 
            working_dir, 
            timeout=10
        )
        with open(os.path.join(working_dir, "fast_correct_examples.json"), "w") as f:
            json.dump(fast_correct_examples, f)
    if os.path.exists(os.path.join(working_dir, "samples.json")):
        samples = json.load(open(os.path.join(working_dir, "samples.json"), "r"))
    else:
        if len(fast_correct_examples) < 10:
            logging.critical(f"Only found {len(fast_correct_examples)} fast_correct_examples, which is less than 10")
        samples = select_n_candidates(fast_correct_examples, n=10, take_top=True)
        with open(os.path.join(working_dir, "samples.json"), "w") as f:
            json.dump(samples, f)    
    # logging.info("Getting all text prompts")
    # problem_id_to_description = get_all_text_prompts()
        # samples = select_n_candidates(fast_correct_examples, n=10, take_top=True)
        # with open(os.path.join(working_dir, "samples.json"), "w") as f, open(os.path.join(working_dir, "fast_correct_examples.json"), "w") as f2:
        #     json.dump(samples, f)
        #     json.dump(fast_correct_examples, f2)
    if os.path.exists(os.path.join(working_dir, "problem_ids_2_all_outputs.json")):
        problem_ids_2_all_outputs = json.load(open(os.path.join(working_dir, "problem_ids_2_all_outputs.json"), "r"))
        problem_ids_2_all_outputs = convert_str_tc_to_int_tc(problem_ids_2_all_outputs)
    else: 
        logging.info(f"Getting all outputs for problem_ids with number of fast_correct_examples: {len(fast_correct_examples)}")
        # problem_ids_2_all_outputs = parallel_get_all_outputs_for_problem_ids(
        problem_ids_2_all_outputs = inner_parallel_get_all_outputs_for_problem_ids(
            samples.keys(),
            fast_correct_examples,
            testcase_root_dir=TESTCASE_ROOT_DIR,
            gcc_opt_flag="-O2", verbose=False, timeout=10 
        )
        problem_ids_2_all_outputs = convert_str_tc_to_int_tc(problem_ids_2_all_outputs)
        with open(os.path.join(working_dir, "problem_ids_2_all_outputs.json"), "w") as f:
            json.dump(problem_ids_2_all_outputs, f)
    ## now start to augment via gpt-3
    # test_build_prompt(fast_correct_examples, list(samples.keys()))
    # test_responses()
    # test_generate_code_descriptipon_pair(fast_correct_examples, list(samples.keys()), max_prompt_length=10000, max_tokens=16000)
    # test_generate_code_only(fast_correct_examples, list(samples.keys()), max_prompt_length=10000, max_tokens=16000)
    logging.info("Generating novel code only")
    novel_code_results = []
    # pbar = tqdm(total=20, desc="Generating novel code only")
    sample_ids = list(problem_ids_2_all_outputs.keys()) * int(args.total_iterations / 10)
    # sample_ids = list(samples.keys())[:1]
    
    # iterations_multiplier = int(args.total_iterations / len(sample_ids))
    # sample_ids = sample_ids * iterations_multiplier
    
    if os.path.exists(os.path.join(results_dir, "generated_results.json")) :#and False: 
        novel_code_results = json.load(open(os.path.join(results_dir, "generated_results.json"), "r"))
    else: 
        check_picklable(locals())
        novel_code_results = parallel_generate_novel_code(fast_correct_examples, 
                                                               problem_ids_2_all_outputs, 
                                                               results_dir, 
                                                               sample_ids,
                                                               max_prompt_length=args.max_prompt_length,
                                                               max_tokens=args.max_tokens,
                                                               top_p=args.top_p,   
                                                               temperature=args.temperature,
                                                               n=args.num_samples,
                                                               generation_strategy=args.generation_strategy)
        logging.info(f"Number of generated programs that differ from the dataset: {len([result for result in novel_code_results if result['is_different']])} / {len(novel_code_results)}")
        with open(os.path.join(results_dir, "generated_results.json"), "w") as f:
            json.dump(novel_code_results, f)
            
    total_cost = sum([result["total_cost"] for result in novel_code_results])
    average_template_len = sum([result["template_len"] for result in novel_code_results]) / len(novel_code_results)
    average_generation_len = sum([result["generation_len"] for result in novel_code_results]) / len(novel_code_results)
    logging.info(f"Total cost of generation: {total_cost:.2f}, average template length: {average_template_len:.2f}, average generation length: {average_generation_len:.2f}")
    
    novel_code_results = [result for result in novel_code_results if result['is_different']]
    gen_id_2_results = {f"generated_{i}": result for i, result in enumerate(novel_code_results)}
        
    total_num_unique_programs, total_num_programs, percent_unique, duplicates = calculate_output_agreement(
        list(samples.keys()), 
        gen_id_2_results,
        verbose=False, timeout=10)
    
    cost_per_unique_program = total_cost / total_num_unique_programs
    
    logging.info(f"Total number of unique programs: {total_num_unique_programs} from {total_num_programs} total generated programs which is {percent_unique:.2f}% and cost per unique program is {cost_per_unique_program:.4f}")
    with open(os.path.join(results_dir_root, "results.txt"), "a+") as f:
        f.write("**********\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Total number of unique programs: {total_num_unique_programs} from {total_num_programs} total generated programs which is {percent_unique:.2f}%" + "\n")
        f.write(f"Total cost of generation: {total_cost:.2f}, average template length: {average_template_len:.2f}, average generation length: {average_generation_len:.2f}\n")
        f.write(f"Cost per unique program: {cost_per_unique_program:.4f}\n")
    
    ## [frozenset({'generated_*'...})] are the duplicates, each frozen set is a set of gen_ids that are duplicates
    ## we want to remove all but one of the duplicates
    
    unique_gen_id_2_results = copy.deepcopy(gen_id_2_results)
    for duplicate_set in duplicates:
        duplicate_set = list(duplicate_set)
        keep_gen_id = duplicate_set[0]
        for duplicate_gen_id in duplicate_set[1:]:
            del unique_gen_id_2_results[duplicate_gen_id]
    with open(os.path.join(results_dir, "unique_generated_results.json"), "w") as f:
        json.dump(unique_gen_id_2_results, f)
    
    
    with open(os.path.join(results_dir, "duplicates.pkl"), "wb") as f:
        pickle.dump(duplicates, f)
    # print(f"Total number of unique programs: {total_unique_programs} from {n_total_programs} total programs which is {total_unique_programs / n_total_programs * 100:.2f}%")
    logging.info(f"Total number of unique programs: {total_num_unique_programs} from {total_num_programs} total programs which is {percent_unique:.2f}%")
    
    ## following used only to test using 2 api calls instead of 1 api call
    
    # logging.info("Generating novel code_description_pair")
    # novel_code_results_code_description_pair = []
    # if os.path.exists(os.path.join(working_dir, "novel_code_results_code_description_pair.json")) and False: 
    #     novel_code_results_code_description_pair = json.load(open(os.path.join(working_dir, "novel_code_results_code_description_pair.json"), "r"))
    # else:
    #     # pbar = tqdm(total=20, desc="Generating novel code_description_pair")
    #     # for _ in range(2):
    #     # _novel_code_results_code_description_pair = parallel_generate_novel_code(fast_correct_examples, problem_ids_2_all_outputs, working_dir, sample_ids, max_prompt_length=10000, generation_strategy="code_description_pair")
    #     novel_code_results_code_description_pair = parallel_generate_novel_code(fast_correct_examples, 
    #                                                                             problem_ids_2_all_outputs, 
    #                                                                             working_dir, 
    #                                                                             sample_ids, 
    #                                                                             max_prompt_length=args.max_prompt_length,
    #                                                                             max_tokens=args.max_tokens,
    #                                                                             top_p=args.top_p,
    #                                                                             temperature=args.temperature,
    #                                                                             generation_strategy="code_description_pair")
    #     # novel_code_results_code_description_pair.extend(_novel_code_results_code_description_pair)
    #         # print(f"Number of novel code_description_pair programs: {len([result for result in novel_code_results_code_description_pair if result['is_different']])} / {len(novel_code_results_code_description_pair)}")
    #         # pbar.set_description(f"No. novel code_description_pair programs: {len([result for result in novel_code_results_code_description_pair if result['is_different']])} / {len(novel_code_results_code_description_pair)}")
    #         # pbar.update(10)
    #     logging.info(f"Number of novel code_description_pair programs: {len([result for result in novel_code_results_code_description_pair if result['is_different']])} / {len(novel_code_results_code_description_pair)}")
    
    # gen_id_2_results = {f"generated_{i}": result for i, result in enumerate(novel_code_results_code_description_pair)}
    # total_num_unique_programs, total_num_programs, percent_unique = calculate_output_agreement(
    #     list(samples.keys()), 
    #     gen_id_2_results,
    #     verbose=False, timeout=10)
    # # print(f"Total number of unique programs: {total_unique_programs} from {n_total_programs} total programs which is {total_unique_programs / n_total_programs * 100:.2f}%")
    
    # logging.info(f"Total number of unique programs: {total_num_unique_programs} from {total_num_programs} total programs which is {percent_unique:.2f}%")
    
    # ## summarize how many is_different from each 
    # # logging.info(f"Number of novel code_description_pair programs: {len([result for result in novel_code_results_code_description_pair if result['is_different']])} / {len(novel_code_results_code_description_pair)}")
    # with open(os.path.join(working_dir, "novel_code_results_code_description_pair.json"), "w") as f:
    #     json.dump(novel_code_results_code_description_pair, f)
    
        
        
                

            
    
if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", type=str, default="/home/alex/Documents/PennPhD/learning2perf/data/data_augmentation/working_dir")
    parser.add_argument("--results_dir_root", type=str, default="/home/alex/Documents/PennPhD/learning2perf/data/data_augmentation/results_dir")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--max_prompt_length", type=int, default=10000)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-16k-0613")
    parser.add_argument("--generation_strategy", type=str, default="code_only")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--total_iterations", type=int, default=10)
    args = parser.parse_args()
    main(args)