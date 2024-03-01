from flask import Flask, request, jsonify
import argparse
import json
import logging
from datetime import datetime
import os
from joblib import Parallel, delayed
import benchmarking
import tempfile
import multiprocessing
import numpy as np
import joblib
from tqdm import tqdm
import contextlib

LOGGING_DIR="/home/logs/"
if not os.path.exists(LOGGING_DIR): 
    os.makedirs(LOGGING_DIR)


logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

# Create a file handler for the log file
start_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
file_handler = logging.FileHandler(os.path.join(LOGGING_DIR, start_date_time + "_gem5_api.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Create a stream handler to print the logs to stdout
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


app = Flask(__name__)


global MANAGER
global QUEUE
global N_CPUS
MANAGER = ...
QUEUE = ...
N_CPUS=... # Will be set in init_globals after parse_args()

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
        

def init_globals(n_workers: int = -1, use_logical_cpus: bool = False): 
    global MANAGER
    global QUEUE 
    global N_CPUS
    
    MANAGER = multiprocessing.Manager()
    QUEUE = MANAGER.Queue()
    if use_logical_cpus: 
        cpu_list = benchmarking.add_logicial_cpus_to_queue(n_workers, QUEUE)
    else: 
        cpu_list = benchmarking.add_physical_cpus_to_queue(n_workers, QUEUE)
    N_CPUS = len(cpu_list)
    print(f"Initialized globals with {N_CPUS} cpus")
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Gem5 API')
    parser.add_argument('--api_key', type=str, help='required API key on initialization for authentication')
    parser.add_argument('--port', type=int, default=706965, help='port number')
    parser.add_argument('--working_dir', type=str, default='/home/working_dir', help='working directory')
    parser.add_argument('--use_logical_cpus',  default=False, action="store_true") 
    parser.add_argument('--workers', type=int, default=-1, help='number of workers, if <0 (e.g. -1) then it uses all available physical cpus')
    parser.add_argument('--threaded',  default=False, action="store_true")
    parser.add_argument('--gem5_acc_threshold', type=float, default=0.95, help="mean threshold where if below this, we do not run gem5")
    parser.add_argument('--debug',  default=False, action="store_true")
    parser.add_argument('--exit_early_on_fail', action="store_true")
    ## gem5 and compilation parameters
    parser.add_argument('--testcases_dir', type=str, help='testcases directory', default="/home/pie-perf/data/codenet/merged_test_cases/")
    parser.add_argument('--cstd', type=str, help='cstd', default='--std=c++17')
    parser.add_argument('--optimization_flag', type=str, help='optimization', default='-O3')
    parser.add_argument('--gem5_dir', type=str, help='path containing gem5 binary and build', default='/home/gem5/build/X86/')
    parser.add_argument('--gem5_script_path', type=str, help='path to gem5 script', default='/home/gem5-skylake-config/gem5-configs/run-se.py')
    parser.add_argument('--cpu_type', type=str, help='cpu type', default='Verbatim')
    parser.add_argument('--path_to_atcoder', type=str, help='path to atcoder', default='/home/ac-library/')
    parser.add_argument('--timeout_seconds_binary', type=int, help='timeout seconds for binary', default=10)
    parser.add_argument('--timeout_seconds_gem5', type=int, help='timeout seconds for gem5', default=120)
    
    
    args = parser.parse_args()
    app.config.update(vars(args))
    return args

def single_submission(code, testcases, problem_id, timing_env, queue, override_flags=""):
    ## TODO -> check if any test cases are missing with hyperfine
    logging.info(f"single_submission for problem {problem_id} with timing_env {timing_env} and testcases {testcases}")
    override_flags = "" if not isinstance(override_flags, str) else override_flags
    result = {}
    cpu_number = queue.get(block=True) if timing_env in ("binary", "both") else None
    logging.info(f"got cpu {cpu_number} in pid {os.getpid()}")
    with tempfile.TemporaryDirectory() as tmpdirname:
        code_path = os.path.join(tmpdirname, 'code.cpp')
        with open(code_path, 'w') as f:
            f.write(code)
        print(f"app cfg cstd {app.config['cstd']} app.config['optimization_flag']: {app.config['optimization_flag']}  override_flags: {override_flags }")
        cflags = app.config['cstd'] + ' ' + app.config['optimization_flag'] + override_flags
        bin_path, accs = benchmarking.compile_and_check_outputs(
            code_path=code_path,
            problem_id=problem_id,
            testcases_dir=app.config['testcases_dir'], 
            timeout=app.config['timeout_seconds_binary'],
            cflags=cflags, 
            testcases=testcases, 
            cpu_number=cpu_number)
        result["compile_success"] = bin_path is not None
        result['accs'] = accs
        mean_accs = np.mean(list(accs.values()))
        logging.info(f"mean_accs: {mean_accs}")
        if mean_accs < app.config["gem5_acc_threshold"]: 
            logging.info(f"mean_accs: {mean_accs} is below threshold {app.config['gem5_acc_threshold']}, skipping gem5")
            if timing_env in ["gem5", "both"]:
                result["gem5"] = {} # return empty dict
            if timing_env in ["binary", "both"]:
                result["binary"] = {} # return empty dict
            return result
        
        if timing_env in ['gem5', 'both']: 
            logging.info(f"running gem5 for problem {problem_id}")
            gem5_results = benchmarking.run_gem5(
                gem5_dir=app.config['gem5_dir'],
                gem5_script_path=app.config['gem5_script_path'],
                cpu_type=app.config['cpu_type'],
                bin_path=bin_path,
                problem_id=problem_id,
                testcases_dir=app.config['testcases_dir'],
                timeout=app.config['timeout_seconds_gem5'],
                testcases=testcases,
                cpu_number=cpu_number, 
                exit_early_on_fail=app.config['exit_early_on_fail'])
            result['gem5'] = gem5_results
        if timing_env in ['binary', 'both']:
            code2results, output = benchmarking.run_hyperfine(
                code_paths=[code_path],
                problem_ids=[problem_id],
                path_to_testcases=app.config['testcases_dir'],
                # TODO: REMOVE THIS HERE
                json_out_path=os.path.join(tmpdirname, 'hyperfine_results.json'),
                test_cases_list=[testcases],
                min_runs_per_test_case=10,
                max_runs_per_test_case=500,
                warmup_runs_per_test_case=5,
                cpu_number=cpu_number, 
                do_sanity_check=True) # TODO: PIN TO CPU
            binary_results = code2results[code_path]
            result["binary"] = binary_results
    queue.put(cpu_number)
    return result


def dual_submission(code_v0, code_v1, testcases, problem_id, timing_env, queue, override_flags_v0="", override_flags_v1=""):
    override_flags_v0 = "" if not isinstance(override_flags_v0, str) else override_flags_v0
    override_flags_v1 = "" if not isinstance(override_flags_v1, str) else override_flags_v1
    result = {}
    cpu_number = queue.get(block=True)
    with tempfile.TemporaryDirectory() as tmpdirname_v0, tempfile.TemporaryDirectory() as tmpdirname_v1:
        code_path_v0 = os.path.join(tmpdirname_v0, 'code.cpp')
        with open(code_path_v0, 'w') as f:
            f.write(code_v0)
        code_path_v1 = os.path.join(tmpdirname_v1, 'code.cpp')
        with open(code_path_v1, 'w') as f:
            f.write(code_v1)

        print(f"app cfg cstd {app.config['cstd']} app.config['optimization_flag']: {app.config['optimization_flag']}  override_flags_v0: {override_flags_v0 }")
        cflags_v0 = app.config['cstd'] + ' ' + app.config['optimization_flag'] + override_flags_v0 
        cflags_v1 = app.config['cstd'] + ' ' + app.config['optimization_flag'] + override_flags_v1
        
        bin_path_v0, accs_v0 = benchmarking.compile_and_check_outputs(
            code_path=code_path_v0,
            problem_id=problem_id,
            testcases_dir=app.config['testcases_dir'], 
            timeout=app.config['timeout_seconds_binary'],
            cflags=cflags_v0, 
            testcases=testcases, 
            cpu_number=cpu_number)
        bin_path_v1, accs_v1 = benchmarking.compile_and_check_outputs(
            code_path=code_path_v1,
            problem_id=problem_id,
            testcases_dir=app.config['testcases_dir'], 
            timeout=app.config['timeout_seconds_binary'],
            cflags=cflags_v1, 
            testcases=testcases, 
            cpu_number=cpu_number)
        result["compile_success_v0"] = bin_path_v0 is not None
        result["compile_success_v1"] = bin_path_v1 is not None
        result['accs_v0'] = accs_v0
        result['accs_v1'] = accs_v1
        if timing_env in ['gem5', 'both']:
            gem5_results_v0 = benchmarking.run_gem5(
                gem5_dir=app.config['gem5_dir'],
                gem5_script_path=app.config['gem5_script_path'],
                cpu_type=app.config['cpu_type'],
                bin_path=bin_path_v0,
                problem_id=problem_id,
                testcases_dir=app.config['testcases_dir'],
                timeout=app.config['timeout_seconds_gem5'],
                testcases=testcases,
                cpu_number=cpu_number, 
                exit_early_on_fail=app.config['exit_early_on_fail'])
            result['gem5_v0'] = gem5_results_v0
            gem5_results_v1 = benchmarking.run_gem5(
                gem5_dir=app.config['gem5_dir'],
                gem5_script_path=app.config['gem5_script_path'],
                cpu_type=app.config['cpu_type'],
                bin_path=bin_path_v1,
                problem_id=problem_id,
                testcases_dir=app.config['testcases_dir'],
                timeout=app.config['timeout_seconds_gem5'],
                testcases=testcases,
                cpu_number=cpu_number, 
                exit_early_on_fail=app.config['exit_early_on_fail'])
            result['gem5_v1'] = gem5_results_v1
        if timing_env in ['binary', 'both']:
            code2results, output = benchmarking.run_hyperfine(
                code_paths=[code_path_v0, code_path_v1],
                problem_ids=[problem_id, problem_id],
                path_to_testcases=app.config['testcases_dir'],
                json_out_path=os.path.join(tmpdirname_v0, 'hyperfine_results.json'),
                test_cases_list=[testcases, testcases],
                min_runs_per_test_case=10,
                max_runs_per_test_case=500,
                warmup_runs_per_test_case=5,
                cpu_number=cpu_number, 
                do_sanity_check=True)
            result["binary_v0"] = code2results[code_path_v0]
            result["binary_v1"] = code2results[code_path_v1]
    queue.put(cpu_number)
    return result


def multiple_single_submissions(code_list, testcases_list, problem_id_list, timing_env, queue, cpus, override_flags_list=None):
    assert len(code_list) == len(testcases_list) == len(problem_id_list) == len(override_flags_list)
    with tqdm_joblib(tqdm(desc="Running multiple single submissions", total=len(code_list))) as progress_bar:
        results = Parallel(n_jobs=cpus, verbose=10, backend="multiprocessing")(delayed(single_submission)(code, testcases, problem_id, timing_env, queue, override_flags) for code, testcases, problem_id, override_flags in zip(code_list, testcases_list, problem_id_list, override_flags_list))
    return results

def multiple_dual_submissions(code_v0_list, code_v1_list, testcases_list, problem_id_list, timing_env, queue, cpus, override_flags_list_v0, override_flags_list_v1):
    assert len(code_v0_list) == len(code_v1_list) == len(testcases_list) == len(problem_id_list) == len(override_flags_list_v0) == len(override_flags_list_v1)
    results = Parallel(n_jobs=cpus, verbose=10, backend="multiprocessing")(delayed(dual_submission)(code_v0, code_v1, testcases, problem_id, timing_env, queue, override_flags_v0, override_flags_v1) for code_v0, code_v1, testcases, problem_id, override_flags_v0, override_flags_v1 in zip(code_v0_list, code_v1_list, testcases_list, problem_id_list, override_flags_list_v0, override_flags_list_v1))
    return results

    
@app.route('/gem5/single_submission', methods=['GET'])
def SingleSubmission(): 
    req = request.get_json()
    if req["api_key"] != app.config["api_key"]:
        return jsonify({"error": "Invalid API key"})
    code = req['code']
    testcases = req['testcases']
    problem_id = req['problem_id']
    timing_env = req['timing_env']
    assert len(testcases) > 0
    assert len(code) > 0
    assert timing_env in ['gem5', 'binary', 'both']
    
    override_flags = req.get('override_flags', "")
    results = single_submission(code, testcases, problem_id, timing_env, QUEUE, override_flags)
    return jsonify(results)

@app.route('/gem5/multiple_single_submissions', methods=['GET'])
def MultipleSubmissions():
    req = request.get_json()
    if req["api_key"] != app.config["api_key"]:
        return jsonify({"error": "Invalid API key"})
    submissions = req['submissions']
    timing_env = req['timing_env']
    code_list = [r['code'] for r in submissions]
    testcases_list = [r['testcases'] for r in submissions]
    problem_id_list = [r['problem_id'] for r in submissions]
    override_flags_list = [r.get('override_flags_list', "") for r in submissions]
    
    assert len(code_list) == len(testcases_list) == len(problem_id_list) == len(override_flags_list)
    assert timing_env in ['gem5', 'binary', 'both']
    assert len(code_list) > 0
    assert len(testcases_list) > 0
    assert len(problem_id_list) > 0
    assert len(override_flags_list) > 0
    assert all([len(code) > 0 for code in code_list])
    assert all([len(testcases) > 0 for testcases in testcases_list])
    
    results = multiple_single_submissions(code_list, testcases_list, problem_id_list, timing_env, QUEUE, N_CPUS, override_flags_list)
    
    return jsonify(results)

@app.route('/gem5/single_submission_pair', methods=['GET'])
def SingleSubmissionPair():
    req = request.get_json()
    if req["api_key"] != app.config["api_key"]:
        return jsonify({"error": "Invalid API key"})
    #assert len(req) == 2
    code_v0 = req['code_v0']
    code_v1 = req['code_v1']
    testcases = req['testcases']
    problem_id = req['problem_id']
    timing_env = req['timing_env']
    assert len(testcases) > 0
    assert len(code_v0) > 0
    assert len(code_v1) > 0
    assert timing_env in ['gem5', 'binary', 'both']
    
    override_flags = req.get('override_flags', "")
    results = dual_submission(code_v0, code_v1, testcases, problem_id, timing_env, QUEUE, override_flags)
    return jsonify(results)

@app.route('/gem5/multiple_submissions_pairs', methods=['GET'])
def MultipleSubmissionsPair():
    req = request.get_json()
    if req["api_key"] != app.config["api_key"]:
        return jsonify({"error": "Invalid API key"})
    submissions_v0 = req['submissions_v0']
    submissions_v1 = req['submissions_v1']
    timing_env = req['timing_env']
    
    code_list_v0 = [r['code'] for r in submissions_v0]
    code_list_v1 = [r['code'] for r in submissions_v1]
    testcases_list = [r['testcases'] for r in submissions_v0]
    problem_id_list = [r['problem_id'] for r in submissions_v0]
    
    override_flags_list_v0 = [r.get('override_flags_list', "") for r in submissions_v0]
    override_flags_list_v1 = [r.get('override_flags_list', "") for r in submissions_v1]
    
    assert len(code_list_v0) == len(testcases_list) == len(problem_id_list) == len(override_flags_list_v0) == len(code_list_v1) == len(override_flags_list_v1)
    assert timing_env in ['gem5', 'binary', 'both']
    assert len(code_list_v0) > 0
    assert len(testcases_list) > 0
    assert len(problem_id_list) > 0
    assert all([len(code) > 0 for code in code_list_v0])
    assert all([len(code) > 0 for code in code_list_v1])
    assert all([len(testcases) > 0 for testcases in testcases_list])
    
    results = multiple_dual_submissions(code_list_v0, code_list_v1, testcases_list, problem_id_list, timing_env, QUEUE, N_CPUS, override_flags_list_v0, override_flags_list_v1)
    return jsonify(results)

@app.route('/gem5/ping', methods=['GET'])
def Ping():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    args = parse_args()
    init_globals(args.workers, args.use_logical_cpus)
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
    
    
    
    

    
    
