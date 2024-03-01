import argparse
import pandas as pd
import shutil
import os
import warnings
import traceback
import logging
import subprocess
import glob
import re
import traceback
import time
import shlex
from typing import Optional, List, Tuple, Dict, Any, Union
import multiprocessing
from collections import defaultdict
import json 
import resource
import re
import ast
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("resource").setLevel(logging.DEBUG)

MAX_VIRTUAL_MEMORY = 10 * 1024 * 1024 * 50  # 500 MB

# from https://gist.github.com/s3rvac/f97d6cbdfdb15c0a32e7e941f7f4a3fa
def limit_virtual_memory():
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, MAX_VIRTUAL_MEMORY * 10))
    
    
def get_accuracy(output: str, ground_truth: str) -> float:
    """
    Compare the output of the code with the ground truth.
    """
    num_correct = 0
    ground_truth_lines = ground_truth.strip().splitlines()
    output_truth_lines = output.strip().splitlines()
    for gen_output, ground_truth_output in zip(output_truth_lines, ground_truth_lines):
        is_corr = gen_output == ground_truth_output
        if not is_corr:
            try:
                gen_output = float(gen_output)
                ground_truth_output = float(ground_truth_output)
                is_corr = abs(gen_output - ground_truth_output) < 1e-3
            except:
                pass
        num_correct += int(is_corr)

    return num_correct / len(ground_truth_lines)

def compile_cpp_code(code_path: str, timeout: int = 30, output_path: str = None, cflags: str = "--std=c++17 -O3", cpu_number: Optional[int] = None) -> str:
    """_summary_

    Args:
        code_path (str): _description_
        output_path (str, optional): _description_
        cflags (str, optional): _description_
    
    Returns:
        str: _description_
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(code_path), f"{os.path.splitext(os.path.basename(code_path))[0]}.out")
    cpu_cmd = f"taskset --cpu-list {cpu_number}" if cpu_number is not None else ""
        
    cmd = shlex.split(cpu_cmd) + ["/usr/bin/g++", code_path, "-o", output_path] + shlex.split(cflags.replace('"', "").replace("'", ""))
    logging.critical(f"Running command: {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
    if p.returncode != 0:
        raise Exception(f"Error compiling code: {code_path} with command: {' '.join(cmd)}, return code: {p.returncode}, stderr: {p.stderr}")
    else: 
        # sometimes there can be latency in the file system, so we wait a bit
        while(not os.path.exists(output_path)):
            time.sleep(0.05)
    return output_path

def exec_bin(bin_path, in_path, timeout, cpu_number=None):
    logging.info(f'executing {bin_path}, with input {in_path}')
    if in_path is not None:
        fh = open(in_path, 'r')
    else: 
        fh = subprocess.DEVNULL
    cmd = [bin_path]
    if cpu_number is not None:
        cmd = ["taskset", "--cpu-list", str(cpu_number)] + cmd
    p = subprocess.run(cmd, capture_output=True, timeout=timeout, stdin=fh, text=True)
    if in_path is not None:
        fh.close()
    return p.returncode, p.stdout, p.stderr

def exec_gem5(gem5_dir, gem5_script_path, cpu_type, bin_path, in_path, stats_out_path, timeout: str = None, cpu_number=None):
    gem5_bin = os.path.join(gem5_dir, 'gem5.opt')
    cmd = shlex.split(f"{gem5_bin} --stats-file={stats_out_path} {gem5_script_path} {cpu_type} {bin_path}")
    if cpu_number is not None:
        cmd = ["taskset", "--cpu-list", str(cpu_number)] + cmd
    if in_path is not None:
        logging.info(f'executing {" ".join(cmd)}, with input {in_path}')
        with open(in_path, 'r') as fh:
            p = subprocess.run(cmd, capture_output=True, timeout=timeout, stdin=fh, text=True)
    else: 
        logging.info(f'executing {" ".join(cmd)}, with no input')
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
    return p.returncode, p.stdout, p.stderr
    
def exec_bin_for_acc(bin_path, in_path, ground_truth_output, timeout=None):
    logging.info(f'executing {bin_path}, with input {in_path}')
    with open(in_path, 'r') as fh:
        p = subprocess.run([bin_path], capture_output=True, timeout=timeout, stdin=fh, text=True)
    if p.returncode != 0:
        raise Exception(f"Error executing code: {bin_path}, return code: {p.returncode}, stderr: {p.stderr.decode('utf-8')}")
    else: 
        return get_accuracy(p.stdout, ground_truth_output)
    
def compile_and_check_outputs(code_path, problem_id, testcases_dir, timeout=None, cflags: str ="--std=c++17 -O3", testcases: List[int] = None, cpu_number=None):
    
    input_output_pairs = {}
    input_paths = glob.glob(os.path.join(testcases_dir, problem_id, f"input.*.txt"))
    for in_path in input_paths:
        tc_no = re.search(r"input\.(\d+)\.txt", in_path).group(1)
        if testcases is not None and int(tc_no) not in testcases and tc_no not in testcases: # allow both int and str
            continue
        out_path = os.path.join(testcases_dir, problem_id, f"output.{tc_no}.txt")
        input_output_pairs[tc_no] = (in_path, out_path)
    logging.info(f"Found {len(input_output_pairs)} testcases for problem: {problem_id} in testcases_dir: {testcases_dir} with testcases: {testcases}")
    try: 
        bin_path = compile_cpp_code(code_path, timeout, cflags=cflags, cpu_number=cpu_number)
        logging.info(f"Compiled {code_path} to {bin_path}")
    except Exception as e:
        return None, {tc_no: 0 for tc_no in input_output_pairs.keys()}
    
    accs = {}    
    
    for tc_no, (in_path, out_path) in input_output_pairs.items():
        with open(out_path, 'r') as fh:
            ground_truth_output = fh.read().strip()
        try:
            acc = exec_bin_for_acc(bin_path, in_path, ground_truth_output, timeout)
            accs[tc_no] = acc
        except Exception as e:
            logging.error(f"Error executing code: {bin_path} with input: {in_path}, error: {e}")
            accs[tc_no] = 0
            
    logging.info(f"bin_path: {bin_path}, accs: {accs}")
            
    return bin_path, accs

def compile_and_check_outputs_multi(
    code_paths, 
    problem_ids, 
    testcases_dir,
    timeout=None,
    cflags: str ="--std=c++17 -O3",
    test_cases_list = None,
    cpu_number=None): 
    if test_cases_list is None:
        test_cases_list = [None for _ in range(len(code_paths))]
    code2results = defaultdict(dict)
    for code_path, problem_id, test_cases in zip(code_paths, problem_ids, test_cases_list):
        bin_path, accs = compile_and_check_outputs(code_path, problem_id, testcases_dir, timeout, cflags, test_cases, cpu_number)
        code2results[code_path]["compile_success"] = bin_path is not None
        code2results[code_path]["bin_path"] = bin_path
        code2results[code_path]["accs"] = accs
    return code2results


def calc_sim_seconds(stats):
    return float(stats["sim_ticks"]) / float(stats["sim_freq"]) # more accurate than sim_seconds


def parse_stats_txt(stats_path):
    with open(stats_path, 'r') as f:
        stats_lines = f.readlines()
    
    stats = {}
    for line in stats_lines:
        if line.strip() == '':
            continue
        if "Begin" in line:
            continue
        if "End" in line:
            continue
        line = re.sub("#.*", "", line).strip() # remove comments
        parts = line.split()
        parts = [part.strip() for part in parts]
        if len(parts) > 2: 
            value = parts[1:]
        elif len(parts) == 2:
            value = parts[1]
        else: 
            logging.warn(f'could not parse line {line}')
            continue
        key = parts[0]
        if isinstance(value, str): 
            try: 
                value = value.replace("%", "").replace("nan", "None").replace("inf", "None").replace("-inf", "None")
                value = ast.literal_eval(value) if value != "None" else None
            except:
                logging.warn(f"could not parse value {value} for key {key}")
        elif isinstance(value, list):
            try: 
                value = [v.replace("%", "").replace("nan", "None").replace("inf", "None").replace("-inf", "None") for v in value]
                value = [ast.literal_eval(v) if v != "None" else None for v in value]
            except:
                logging.warn(f"could not parse value {value} for key {key}")
        stats[key] = value
    stats["sim_seconds_precise"] = calc_sim_seconds(stats)
    return stats
     

def run_gem5(gem5_dir, gem5_script_path, cpu_type, bin_path, problem_id, testcases_dir, timeout, testcases: List[int] = None, cpu_number=None, exit_early_on_fail=True):
    input_paths = glob.glob(os.path.join(testcases_dir, problem_id, f"input.*.txt"))
    tc_2_in_path = {}
    logging.info(f"Found {len(input_paths)} total testcases for problem: {problem_id} in testcases_dir: {testcases_dir} with testcases: {testcases}")
    for in_path in input_paths:
        tc_no = int(re.search(r"input\.(\d+)\.txt", in_path).group(1))
        if testcases is not None and str(tc_no) not in testcases and tc_no not in testcases:
            continue
        tc_2_in_path[tc_no] = in_path
    logging.info(f"Found {len(tc_2_in_path)} testcases to actually run for problem: {problem_id} in testcases_dir: {testcases_dir} with testcases: {testcases}")
    tc_2_results = {}
    any_incorrect_or_timeout = False
    logging.critical(f"Running {bin_path} on testcases: {tc_2_in_path.keys()}")
    for tc_no, in_path in tc_2_in_path.items():
        # logging.critical(f"Running {bin_path} on testcase {tc_no} with input {in_path}")
        #### TOOD: MAKE SURE ALL CODE/BINARIES ARE IN UNIQUE DIRECTORIES
        stats_out_path = os.path.splitext(bin_path)[0] + f".{tc_no}.txt"
        if exit_early_on_fail and any_incorrect_or_timeout:
            tc_2_results[tc_no] = {"success": False, "error": "Previous testcase was incorrect or timed out, so skipping this testcase",
                                   "stats": None, "stdout": None, "stderr": None, "time": None} 
        else: 
            try: 
                returncode, stdout, stderr = exec_gem5(gem5_dir, gem5_script_path, cpu_type, bin_path, in_path, stats_out_path, timeout, cpu_number=cpu_number)
                if returncode != 0:
                    tc_2_results[tc_no] = {"success": False, "error": f"Error executing code: {bin_path}, return code: {returncode}, stderr: {stderr.decode('utf-8')}", 
                                        "stats": None, "stdout": stdout, "stderr": stderr, "time": None}
                    any_incorrect_or_timeout = True
                else: 
                    tc_2_results[tc_no] = {"success": True, "error": None, "stats": parse_stats_txt(stats_out_path), "stdout": stdout, "stderr": stderr, "time": parse_stats_txt(stats_out_path)["sim_seconds_precise"]}
            except Exception as e:
                traceback_err = traceback.format_exc()
                tc_2_results[tc_no] = {"success": False, "error": f"Error executing code: {bin_path}, error: {e}, traceback: {traceback_err}", 
                                        "stats": None, "stdout": None, "stderr": None, "time": None}
                any_incorrect_or_timeout = True
    return tc_2_results     


def run_gem5_multi(gem5_dir, gem5_script_path, cpu_type, bin_paths, problem_ids, testcases_dir, timeout, test_cases_list: List[int] = None, cpu_number=None, exit_early_on_fail=True):
    if test_cases_list is None:
        test_cases_list = [None for _ in range(len(bin_paths))]
    bin2results = defaultdict(dict)
    for bin_path, problem_id, test_cases in zip(bin_paths, problem_ids, test_cases_list):
        bin2results[bin_path] = run_gem5(gem5_dir, gem5_script_path, cpu_type, bin_path, problem_id, testcases_dir, timeout, test_cases, cpu_number, exit_early_on_fail)
    return bin2results

#### hyperfine

FSTREAM_HEADER="#include <fstream>" # for redirecting io

CPP_HEADERS=[FSTREAM_HEADER]

def make_redirect_io_cpp(testcase_path, output_path=None): 
    lines = f"\nstd::ifstream cin(\"{testcase_path}\");\n"
    if output_path: 
        lines = lines + f"std::ofstream cout(\"{output_path}\");\n\n"
    return lines

def add_headers_cpp(code_str): 
    for header in CPP_HEADERS:
        if header not in code_str:
            code_str = header + "\n" + code_str    
    return code_str


def insert_io_redirects_cpp(code_str, path_to_testcases, path_to_outputs=None): 
    import re
    ## match all whitespace after main and include that in the match greedy
    m = re.search("main(\s*)[^\{}]*{", code_str)
    if m is None:
        raise ValueError("No main function found")
    insert_idx = m.end()
    io_redirects = make_redirect_io_cpp(path_to_testcases, path_to_outputs)
    return code_str[:insert_idx] + io_redirects + code_str[insert_idx:]


def redirect_cpp_io(code_str, path_to_testcases, path_to_outputs=None): 
    code_str = add_headers_cpp(code_str)
    code_str = insert_io_redirects_cpp(code_str, path_to_testcases, path_to_outputs)
    return code_str


def redirect_cpp_io_file(code_path, stdin_path, stdout_path=None, new_code_dir=None): 
    input_basename = os.path.splitext(os.path.basename(stdin_path))[0].replace(".", "_")
    if new_code_dir is None:
        new_code_dir = os.path.dirname(code_path)
    if stdout_path is None:
        basename = os.path.splitext(os.path.basename(code_path))[0]
        stdout_path = os.path.join(new_code_dir, f"{basename}_{input_basename}.stdout")        
    with open(code_path, "r") as f:
        code_str = f.read()
    code_str = redirect_cpp_io(code_str, stdin_path, stdout_path)
    new_code_path = os.path.join(new_code_dir, f"redirected_{input_basename}_{os.path.basename(code_path)}")
    with open(new_code_path, "w") as f:
        f.write(code_str)
    return new_code_path, stdout_path


def redirect_cpp_io_and_compile(code_path, stdin_path, cpu_number=None, new_code_dir=None, stdout_path=None, cflags="--std=c++17 -O3"): 
    new_code_path, stdout_path = redirect_cpp_io_file(code_path, stdin_path, new_code_dir, stdout_path)
    new_binary_path = compile_cpp_code(new_code_path, cpu_number=cpu_number, cflags=cflags)
    return new_binary_path, new_code_path, stdout_path

    
## physical / logical cpu management

def get_physical_cpu_list():
    cmd = " grep -E '^processor|^physical id|^core id' /proc/cpuinfo "
    output = os.popen(cmd).read()
    output = output.split("processor")
    output = [x for x in output if x]
    physical2logical = defaultdict(list)
    n_logical = 0
    for cpu_info in output:
        logical_id = re.search("(?<=\t: )\d+", cpu_info).group(0)
        physical_id = re.search("(?<=core id\t\t: )\d+", cpu_info).group(0)
        physical2logical[int(physical_id)].append(int(logical_id))
        n_logical += 1
    n_physical = len(physical2logical)
    from pprint import pformat
    logging.info(f"Physical CPU (n={n_physical}) to Logical CPU (n={n_logical}) mapping:")
    logging.info(pformat(sorted(dict(physical2logical).items(), key=lambda x: int(x[0]))))
    unique_logical_ids = []
    for physical_id, logical_ids in physical2logical.items():
        unique_logical_ids.append(logical_ids[0])
    logging.info(f"The set of logical ids available for use (n={len(unique_logical_ids)}):")
    logging.info(unique_logical_ids)
    return unique_logical_ids

def add_logicial_cpus_to_queue(num_processes, queue):
    highest_num_processes = multiprocessing.cpu_count() 
    if num_processes < 0: 
        num_processes = highest_num_processes
    else: 
        if num_processes > highest_num_processes:
            raise ValueError(f"num_processes {num_processes} is greater than the highest available cpu: {highest_num_processes}.")
    available_cpus = list(range(num_processes))
    if len(available_cpus) > 2: 
        available_cpus = available_cpus[:-2]
    else: 
        logging.warning(f"there are fewer than 3 logical CPUs which is not recommended")
    for cpu_id in available_cpus:
        queue.put(cpu_id)
    logging.info(f"List of cpus to be used: {available_cpus}")
    return available_cpus

def add_physical_cpus_to_queue(num_processes, queue):
    available_cpus = [i for i in get_physical_cpu_list() if i >= 0]
    if len(available_cpus) > 2: 
        available_cpus = available_cpus[:-2]
    else: 
        logging.warning(f"there are fewer than 3 physical CPUs which is not recommended")
    if num_processes < 0: 
        num_processes = len(available_cpus)
    elif len(available_cpus) < num_processes:
        raise ValueError(f"Only {len(available_cpus)} available cpus, but {num_processes} processes requested; the set of available cpus is {available_cpus}")
    for cpu_id in available_cpus[:num_processes]:
        queue.put(cpu_id)
    logging.info(f"List of cpus to be used: {available_cpus[:num_processes]}")
    return available_cpus

def run_benchmark(args, json_output_path, timeout_seconds: int = 60) -> Union[str, None]:
    try: 
        logging.info(f"Running {' '.join(args)}")
        proc = subprocess.Popen(
            args, 
            preexec_fn=limit_virtual_memory,
            # stderr=subprocess.DEVNULL, 
            # stdout=subprocess.DEVNULL
        )
        output = proc.communicate(timeout=timeout_seconds)[0]
        if os.path.exists(json_output_path):
            results = json.load(open(json_output_path)).get("results", [])
            return results, output
        else:
            return None, output
    except subprocess.TimeoutExpired: 
        logging.warning(f"Timeout for {args}")
        _kill(proc.pid)  # type: ignore
        return None, f"Timeout after {timeout_seconds} seconds"
    except json.decoder.JSONDecodeError: 
        logging.warning(f"JSONDecodeError for {args}")
        return None, f"JSONDecodeError"
    except KeyboardInterrupt as e:
        _kill(proc.pid)  # type: ignore
        raise e

    
def run_hyperfine(code_paths: List[str], 
                   problem_ids: List[str], 
                   path_to_testcases: str,
                   json_out_path: str, # TODO REMOVE json_out_path
                   test_cases_list: List[int] = None,
                   min_runs_per_test_case: int = None, 
                   max_runs_per_test_case: int = None,
                   strict_runs_per_test_case: bool = False,
                   warmup_runs_per_test_case: int = 5,
                   cpu_number: int = None, 
                   do_sanity_check: bool = False, 
                   cflags: str = "--std=c++17 -O3"):
    """
    will benchmark all in 1 json / 1 run of hyperfine, all on the same cpu
    """
    
    ### TODO: need to change to handle compilation errors and timeouts
    
    code2benchmarks = defaultdict(list)
    benchmark2code = {}
    code2results = defaultdict(dict)
    code2testcases = defaultdict(list)
    if test_cases_list is None: 
        test_cases_list = [None] * len(code_paths)
    for code_path, problem_id, test_case_list in zip(code_paths, problem_ids, test_cases_list):
        problem_dir = os.path.join(path_to_testcases, problem_id)
        testcases_paths = glob.glob(os.path.join(problem_dir, "input.*.txt"))
        if test_case_list is not None:
            testcases_paths = [t for t in testcases_paths if int(re.search("(?<=input\.)\d+", t).group(0)) in test_case_list]
        test_case_numbers = [int(re.search("(?<=input\.)\d+", t).group(0)) for t in testcases_paths]
        code2testcases[code_path] = test_case_numbers
        for testcase_path in testcases_paths:
            bin_redirect, code_redirect, _ = redirect_cpp_io_and_compile(code_path, 
                                                                         testcase_path, 
                                                                         cpu_number=cpu_number, 
                                                                         cflags=cflags)
            code2benchmarks[code_path].append(bin_redirect)
            benchmark2code[bin_redirect] = code_path
    
    cmds = " ".join([bin_redirect for bin_redirects in code2benchmarks.values() for bin_redirect in bin_redirects])
    n_cmds = len(cmds.split(" "))
    if strict_runs_per_test_case:
        assert min_runs_per_test_case is not None 
        runs_str = f" --runs {min_runs_per_test_case}"
    else: 
        runs_str = ""
        if min_runs_per_test_case is not None: 
            runs_str += f" --min-runs {min_runs_per_test_case}"
        if max_runs_per_test_case is not None:
            runs_str += f" --max-runs {max_runs_per_test_case}"
    if warmup_runs_per_test_case is not None:
        runs_str += f" --warmup {warmup_runs_per_test_case}"
    
    cmd_benchmark = (
        f"hyperfine {runs_str} -N {cmds}  --export-json {json_out_path} "
    )
    
    if cpu_number is not None:
        cmd_benchmark = f"taskset --cpu-list {cpu_number} {cmd_benchmark}"
        
    if do_sanity_check: 
        SANITY_CHECK_TIMEOUT = 1.5 * n_cmds
        cmd_sanity_check = cmd_benchmark.replace(runs_str, f" --runs 2 --warmup 1 ") 
        p = subprocess.run(shlex.split(cmd_sanity_check), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=SANITY_CHECK_TIMEOUT, encoding="utf-8")
        if p.returncode != 0:
            return None, f"Sanity check failed for {cmd_sanity_check}: {p.stderr}"
    results, output = run_benchmark(shlex.split(cmd_benchmark), json_out_path)

    for result in results: 
        command = result["command"]
        tc_no = int(re.search("(?<=input\_)\d+", command).group(0))
        code2results[benchmark2code[command]][tc_no] = result
    for bin, code in benchmark2code.items():
        results = code2results[code]
        missing_tcs = set(code2testcases[code]) - set(results.keys())
        for tc_no in missing_tcs:
            results[tc_no] = None
    return code2results, output
        
        
        
    
