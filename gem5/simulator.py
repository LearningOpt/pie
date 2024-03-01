import docker 

import secrets
import string
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Union
from collections import defaultdict
import requests
import time


import os
import tarfile
import docker
import io 
import shutil
from pprint import pprint
import threading
import pdb
import gem5
import inspect
import multiprocessing
import subprocess
import shlex

GEM5_DIR_PATH=os.path.dirname(inspect.getfile(gem5))


DOCKERFILE_TEMPLATE = """
FROM alexshypula/gem5-skylake:api

WORKDIR /home/working_dir

ENV PYTHONPATH=/home/gem5-skylake-config/gem5-configs/system/

USER gem5

COPY benchmarking.py /home/working_dir/benchmarking.py
COPY gem5_api.py /home/working_dir/gem5_api.py

"""

import logging 
logging.basicConfig(level=logging.INFO)

from rich.progress import Progress

# https://stackoverflow.com/questions/65896588/how-to-capture-docker-pull-status
# Show task progress (red for download, green for extract)
def show_progress(line, progress, tasks):
    if line['status'] == 'Downloading':
        id = f'[red][Download {line["id"]}]'
    elif line['status'] == 'Extracting':
        id = f'[green][Extract  {line["id"]}]'
    else:
        # skip other statuses
        return

    if id not in tasks.keys():
        tasks[id] = progress.add_task(f"{id}", total=line['progressDetail']['total'])
    else:
        progress.update(tasks[id], completed=line['progressDetail']['current'])

def image_pull(client, image_name, stream=True): 
    logging.info(f'Pulling image: {image_name}')
    tasks = {}
    with Progress() as progress:
        # client = docker.from_env()
        resp = client.api.pull(image_name, stream=True, decode=True)
        if stream:
            for line in resp:
                show_progress(line, progress, tasks)


def docker_copy_to(src, dst, client): 
    name, dst = dst.split(':')
    container = client.containers.get(name)

    os.chdir(os.path.dirname(src))
    srcname = os.path.basename(src)
    tar = tarfile.open(src + '.tar', mode='w')
    try:
        tar.add(srcname)
    finally:
        tar.close()

    data = open(src + '.tar', 'rb').read()
    container.put_archive(os.path.dirname(dst), data)


def make(*args, **kwargs):
    return PieEnvironment(*args, **kwargs)

def generate_api_key(length=256):
    alphabet = string.digits + string.ascii_letters 
    api_key = ''.join(secrets.choice(alphabet) for _ in range(length))
    return api_key
    
def parse_submission_result(result: Union[List[Dict[str, Any]], Dict[str, Any]]):
    if isinstance(result, list):
        return [_parse_submission(r) for r in result]
    else:
        return _parse_submission(result)
    
def _parse_submission(result: Dict[str, Any]):
    # pprint(f"Result: {result}")
    pprint(f"Keys: {result.keys()}")
    if any(["v0" in key for key in result.keys()]):
        # return _parse_submission_pair(result)
        return PiePairResult.from_dict(result)
    else: 
        # return _parse_single_submission(result)
        return PieSingleResult.from_dict(result)
    

@dataclass
class PieResult: 
    def to_json(self):
        return json.dumps(self.to_dict())
        

@dataclass
class PieSingleResult(PieResult):
    compilation: bool
    accs: Dict[str, float]
    mean_acc: float
    agg_runtime: float
    agg_stdev: float
    tc2time: Dict[str, float]
    tc2success: Dict[str, bool]
    tc2stats: Dict[str, List[float]]
    
    agg_runtime_binary: float = None
    agg_stdev_binary: float = None
    tc2time_binary: Dict[str, float] = None
    tc2success_binary: Dict[str, bool] = None
    tc2stats_binary: Dict[str, List[float]] = None
    
    @staticmethod
    def from_dict(result: Dict[str, Any]):
        parsed_result = _parse_single_submission(result)
        return PieSingleResult(**parsed_result)
    
    def to_dict(self):
        result = {}
        result["compile_success"] = self.compile_success
        result["accs"] = self.accs
        result["mean_acc"] = self.mean_acc
        result["agg_runtime"] = self.agg_runtime
        result["agg_stdev"] = self.agg_stdev
        result["tc2time"] = self.tc2time
        result["tc2success"] = self.tc2success
        result["tc2stats"] = self.tc2stats
        return result
    
    @staticmethod
    def from_json(json_str: str):
        return PieSingleResult.from_dict(json.loads(json_str))
    
    
@dataclass
class PiePairResult(PieResult):
    compilation_v0: bool
    compilation_v1: bool
    accs_v0: Dict[str, float]
    accs_v1: Dict[str, float]
    mean_acc_v0: float
    mean_acc_v1: float
    tc2time_v0: Dict[str, float]
    tc2time_v1: Dict[str, float]
    agg_runtime_v0: float
    agg_runtime_v1: float
    agg_stdev_v0: float
    agg_stdev_v1: float
    tc2success_v0: Dict[str, bool]
    tc2success_v1: Dict[str, bool]
    tc2stats_v0: Dict[str, List[float]]
    tc2stats_v1: Dict[str, List[float]]
    
    
    agg_runtime_binary_v0: float = None
    agg_runtime_binary_v1: float = None
    agg_stdev_binary_v0: float = None
    agg_stdev_binary_v1: float = None
    tc2time_binary_v0: Dict[str, float] = None
    tc2time_binary_v1: Dict[str, float] = None
    tc2success_binary_v0: Dict[str, bool] = None
    tc2success_binary_v1: Dict[str, bool] = None
    tc2stats_binary_v0: Dict[str, List[float]] = None
    tc2stats_binary_v1: Dict[str, List[float]] = None
    
    
    @staticmethod
    def from_dict(result: Dict[str, Any]):
        parsed_result = _parse_submission_pair(result)
        return PiePairResult(**parsed_result)
    
    def to_dict(self):
        result = {}
        result["compilation_v0"] = self.compilation_v0
        result["compilation_v1"] = self.compilation_v1
        result["accs_v0"] = self.accs_v0
        result["accs_v1"] = self.accs_v1
        result["mean_acc_v0"] = self.mean_acc_v0
        result["mean_acc_v1"] = self.mean_acc_v1
        result["tc2time_v0"] = self.tc2time_v0
        result["tc2time_v1"] = self.tc2time_v1
        result["agg_runtime_v0"] = self.agg_runtime_v0
        result["agg_runtime_v1"] = self.agg_runtime_v1
        result["agg_stdev_v0"] = self.agg_stdev_v0
        result["agg_stdev_v1"] = self.agg_stdev_v1
        result["tc2success_v0"] = self.tc2success_v0
        result["tc2success_v1"] = self.tc2success_v1
        result["tc2stats_v0"] = self.tc2stats_v0
        result["tc2stats_v1"] = self.tc2stats_v1
        result["agg_runtime_binary_v0"] = self.agg_runtime_binary_v0
        result["agg_runtime_binary_v1"] = self.agg_runtime_binary_v1
        result["agg_stdev_binary_v0"] = self.agg_stdev_binary_v0
        result["agg_stdev_binary_v1"] = self.agg_stdev_binary_v1
        result["tc2time_binary_v0"] = self.tc2time_binary_v0
        result["tc2time_binary_v1"] = self.tc2time_binary_v1
        result["tc2success_binary_v0"] = self.tc2success_binary_v0
        result["tc2success_binary_v1"] = self.tc2success_binary_v1
        result["tc2stats_binary_v0"] = self.tc2stats_binary_v0
        result["tc2stats_binary_v1"] = self.tc2stats_binary_v1
        
        return result
    
    @staticmethod
    def from_json(json_str: str):
        return PiePairResult.from_dict(json.loads(json_str))
    
    
def _parse_single_submission(result: Dict[str, Any]):
    parsed_result = {}
    compilation = result["compile_success"]
    accs = result["accs"]
    mean_acc = np.mean([acc for acc in accs.values()])
    
    parsed_result["compilation"] = compilation
    parsed_result["accs"] = accs
    parsed_result["mean_acc"] = mean_acc
    
    tc2time = {}
    agg_runtime = 0
    tc2success = {} 
    tc2stats = {}
    # pdb.set_trace()
    
    if "gem5" in result.keys():
        
        gem5_result = result["gem5"]
        if gem5_result is None or gem5_result == {}:
            agg_runtime = np.inf
        else:
            for tc_no, tc_result in gem5_result.items():
                tc_no = int(tc_no)
                tc2success[tc_no] = tc_result["success"]
                if tc2success[tc_no]: 
                    stats =  tc_result["stats"]
                    tc2stats[tc_no] = stats
                    tc2time[tc_no] = stats["sim_seconds_precise"]
                else: 
                    tc2time[tc_no] = np.inf
                agg_runtime += tc2time[tc_no]
        
        parsed_result["agg_runtime"] = agg_runtime
        parsed_result["tc2time"] = tc2time
        parsed_result["tc2success"] = tc2success
        parsed_result["agg_stdev"] = 0 
        parsed_result["tc2stats"] = tc2stats
        
    if "binary" in result.keys():
        tc2time = {}
        agg_runtime = 0
        tc2success = {} 
        tc2stats = {}
        # import pdb; pdb.set_trace()
        key_suffix = "_binary" if "gem5" in result.keys() else ""
        binary_result = result["binary"]
        shortest_len = float("inf")
        for tc_no, tc_result in binary_result.items():
            tc_no = int(tc_no)
            if tc_result is None:
                tc2success[tc_no] = False
                tc2time[tc_no] = np.inf
                tc2stats[tc_no] = None
                shortest_len = 0
            else:
                tc2success[tc_no] = True
                tc2time[tc_no] = tc_result["mean"]
                tc2stats[tc_no] = tc_result["times"]
                shortest_len = min(shortest_len, len(tc_result["times"]))
        if shortest_len == 0:
            agg_runtime = np.inf
            agg_stdev = np.inf
        else:
            agg_runtimes = np.stack([stat[:shortest_len] for stat in tc2stats.values()], axis=0).sum(axis=0)
            agg_runtime = agg_runtimes.mean()
            agg_stdev = agg_runtimes.std()
            
        parsed_result["agg_runtime" + key_suffix] = agg_runtime
        parsed_result["agg_stdev" + key_suffix] = agg_stdev
        parsed_result["tc2time" + key_suffix] = tc2time
        parsed_result["tc2success" + key_suffix] = tc2success
        parsed_result["tc2stats" + key_suffix] = tc2stats
    
    return parsed_result
    
    
def _parse_submission_pair(result: Dict[str, Any]):
    parsed_result = {}
    compilation_v0 = result["compile_success_v0"]
    compilation_v1 = result["compile_success_v1"]
    accs_v0 = result["accs_v0"]
    accs_v1 = result["accs_v1"]
    mean_acc_v0 = np.mean([acc for acc in accs_v0.values()])
    mean_acc_v1 = np.mean([acc for acc in accs_v1.values()])
    
    parsed_result["compilation_v0"] = compilation_v0
    parsed_result["compilation_v1"] = compilation_v1
    parsed_result["accs_v0"] = accs_v0
    parsed_result["accs_v1"] = accs_v1
    parsed_result["mean_acc_v0"] = mean_acc_v0
    parsed_result["mean_acc_v1"] = mean_acc_v1
    
  
    
    parsed_result["tc2time_v0"] = {}
    parsed_result["tc2time_v1"] = {}
    parsed_result["agg_runtime_v0"] = 0
    parsed_result["agg_runtime_v1"] = 0
    parsed_result["tc2success_v0"] = {}
    parsed_result["tc2success_v1"] = {}
    parsed_result["tc2stats_v0"] = {}
    parsed_result["tc2stats_v1"] = {}
    
    if "gem5_v0" in result.keys():
        if "gem5_v1" not in result.keys():
            raise ValueError("gem5_v0 is in result.keys() but gem5_v1 is not")
        
        gem5_v0_result = result["gem5_v0"]
        gem5_v1_result = result["gem5_v1"]
        for gem5_result, key_suffix in [(gem5_v0_result, "_v0"), (gem5_v1_result, "_v1")]:
            
            agg_runtime = 0 
            for tc_no, tc_result in gem5_result.items(): 
                tc_no = int(tc_no)
                parsed_result["tc2success" + key_suffix][tc_no] = tc_result["success"]
                if tc_result["success"]: 
                    stats = tc_result["stats"]
                    parsed_result["tc2stats" + key_suffix][tc_no] = stats
                    parsed_result["tc2time" + key_suffix][tc_no] = stats["sim_seconds_precise"]
                else: 
                    parsed_result["tc2time" + key_suffix][tc_no] = np.inf
                agg_runtime += parsed_result["tc2time" + key_suffix][tc_no]
            
            parsed_result["agg_runtime" + key_suffix] = agg_runtime
            parsed_result["agg_stdev" + key_suffix] = 0
    # import pdb; pdb.set_trace()
    if "binary_v0" in result.keys():
        
        if "binary_v1" not in result.keys():
            raise ValueError("binary_v0 is in result.keys() but binary_v1 is not")
            
        binary_v0_result = result["binary_v0"]
        binary_v1_result = result["binary_v1"]
        for binary_result, key_suffix in [(binary_v0_result, "_v0"), (binary_v1_result, "_v1")]:
            
            if "gem5_v0" in result.keys():
                key_suffix = "_binary" + key_suffix
                parsed_result["agg_runtime" + key_suffix] = 0
                parsed_result["agg_stdev" + key_suffix] = 0
                parsed_result["tc2time" + key_suffix] = {}
                parsed_result["tc2success" + key_suffix] = {}
                parsed_result["tc2stats" + key_suffix] = {}
        
            shortest_len = float("inf")
            for tc_no, tc_result in binary_result.items():
                tc_no = int(tc_no)
                if tc_result is None:
                    parsed_result["tc2success" + key_suffix][tc_no] = False
                    parsed_result["tc2time" + key_suffix][tc_no] = np.inf
                    parsed_result["tc2stats" + key_suffix][tc_no] = None
                    shortest_len = 0
                else:
                    parsed_result["tc2success" + key_suffix][tc_no] = True
                    parsed_result["tc2time" + key_suffix][tc_no] = tc_result["mean"]
                    parsed_result["tc2stats" + key_suffix][tc_no] = tc_result["times"]
                    shortest_len = min(shortest_len, len(tc_result["times"]))
            if shortest_len == 0:
                parsed_result["agg_runtime" + key_suffix] = np.inf
                parsed_result["agg_stdev" + key_suffix] = np.inf
            else:
                tc2stats = parsed_result["tc2stats" + key_suffix]
                agg_runtimes = np.stack([stat[:shortest_len] for stat in tc2stats.values()], axis=0).sum(axis=0)
                parsed_result["agg_runtime" + key_suffix] = agg_runtimes.mean()
                parsed_result["agg_stdev" + key_suffix] = agg_runtimes.std()
            
    return parsed_result
        
    
class PieEnvironment: 
    image = "alexshypula/gem5-skylake:api"
    supported_modes = ("gem5", "binary", "both") 
    api_key=None
    container = None
    stream_thread = None
    
    def __init__(self, 
                 arch: str = 'X86-skylake',
                 cpuset_cpus: str = None, 
                 working_dir: str = '/home/working_dir',
                 use_logical_cpus: bool = False,
                 workers: int = -1,
                 threaded: bool = False,
                 gem5_acc_threshold: float = 0.95,
                 port: int = 4000,
                #  port: int = 4000,
                 cstd: str = '--std=c++17', 
                 optimization_flag: str = '-O3', 
                 cpu_type: str = 'Verbatim', 
                 timeout_seconds_binary: int = 10, 
                 timeout_seconds_gem5: int = 60, 
                 api_key: str = None, 
                 verbose: bool = False, 
                 do_run_without_container: bool = False, 
                 exit_early_on_fail: bool = True): 
        
        if arch != 'X86-skylake':
            raise NotImplementedError(f"Architecture {arch} not supported, only X86-skylake is supported")
        
        self.arch = arch
        self.cpuset_cpus = cpuset_cpus
        self.working_dir = working_dir
        self.use_logical_cpus = use_logical_cpus
        self.workers = workers  
        self.threaded = threaded
        self.gem5_acc_threshold = gem5_acc_threshold
        self.port = port
        self.cstd = cstd
        self.optimization_flag = optimization_flag
        self.cpu_type = cpu_type
        self.timeout_seconds_binary = timeout_seconds_binary
        self.timeout_seconds_gem5 = timeout_seconds_gem5
        self.verbose = verbose
        self.do_run_without_container = do_run_without_container
        self.child_process = None # for use with run_without_container
        self.exit_early_on_fail = exit_early_on_fail
        ## TODO: allow a flag to short-circuit evaluation when we get a timeout 
        
        if api_key is None:
            api_key = generate_api_key()
        self.api_key = api_key
        
        
        if self.do_run_without_container: 
            self.run_without_container()
        else: 
            self.client = docker.from_env()
            self.setup()
            self.run()
        self.wait_for_connection(timeout=30)
        
    def setup(self):
        ## check if image exists
        if self.image not in [tag for image in self.client.images.list() for tag in image.tags]:
            # self.client.images.pull(self.image)
            image_pull(client=self.client, image_name=self.image, stream=True)
            
     
    def run(self): 
        """
        Default run in a ***separate*** docker container
        """
        
        arch_arg = self._get_arch_arg()
        
        command = self.build_gem5_command(arch_arg)
        
        dockerfile = DOCKERFILE_TEMPLATE + "\n" + "EXPOSE " + str(self.port) + "\n" + "CMD " + " ".join(command)
        
        assert not os.path.exists(f"{GEM5_DIR_PATH}/Dockerfile")
        with open(f"{GEM5_DIR_PATH}/Dockerfile", "w") as f:
            f.write(DOCKERFILE_TEMPLATE)
        print("Dockerfile written")
        print("Building docker image")
        img, build_logs = self.client.images.build(path=GEM5_DIR_PATH, tag=self.image + "_bakery", rm=True)
        # https://stackoverflow.com/questions/43540254/how-to-stream-the-logs-in-docker-python-api
        for chunk in build_logs:
            if 'stream' in chunk:
                for line in chunk['stream'].splitlines():
                    print(line)
        if img is None:
            raise Exception("Docker image build failed")
        
        os.remove(f"{GEM5_DIR_PATH}/Dockerfile")
        print("Dockerfile removed")
        
        
        if self.cpuset_cpus is not None:
            container = self.client.containers.run(img.id, command=" ".join(command),
                                                   detach=True, ports={ self.port: self.port}, cpuset_cpus=self.cpuset_cpus, publish_all_ports=True)
            # container = self.client.containers.run(img.id, detach=True, ports={4000: self.port}, cpuset_cpus=self.cpuset_cpus)
        else:
            container = self.client.containers.run(img.id, command=" ".join(command),
                                                   detach=True, ports={ self.port: self.port}, publish_all_ports=True)
            # container = self.client.containers.run(img.id, detach=True, ports={4000: self.port})
        print(f"Attempted to run container with command {' '.join(command)}")
        print(f"Container with status {container.status} on port {self.port}, container name {container.name}")
        print(f"Container logs: {container.logs().decode('utf-8')}")
        self.container = container
        # max_time = 60
        if self.verbose: 
            self.start_stream_thread()

    def _get_arch_arg(self):
        if self.arch == 'X86-skylake': 
            arch_arg = '--gem5_script_path' + ' ' + '/home/gem5-skylake-config/gem5-configs/run-se.py'
        else:
            raise NotImplementedError(f"Architecture {self.arch} not supported")
        return arch_arg
        ## exec command
        
        # docker_copy_to("/home/alex/Documents/PennPhD/learning2perf/gem5/benchmarking.py", f"{self.container.name}:/home/working_dir/benchmarking.py", client=self.client)
        # docker_copy_to("/home/alex/Documents/PennPhD/learning2perf/gem5/gem5_api.py", f"{self.container.name}:/home/working_dir/gem5_api.py", client=self.client)
        # print("Copied files to container")
        
        # self.container.exec_run(" ".join(command), detach=True)
        # print(f"Command executed: {' '.join(command)}")
        
        # if self.cpuset_cpus is not None:
        #     self.client.containers.run(self.image, command, detach=True, ports={4000: self.port}, cpuset_cpus=self.cpuset_cpus)
        # else:
        #     self.client.containers.run(self.image, command, detach=True, ports={4000: self.port})

    def build_gem5_command(self, arch_arg):
        command = ["python3 /home/working_dir/gem5_api.py",
                    arch_arg, f"--port {self.port}", f"--cstd='{self.cstd}'", f"--working_dir {self.working_dir}",
                    f"--workers {self.workers}",
                    f"--gem5_acc_threshold {self.gem5_acc_threshold}", f"--api_key {self.api_key}",
                    f"--optimization_flag='{self.optimization_flag}'", f"--cpu_type {self.cpu_type}",
                    f"--timeout_seconds_binary {self.timeout_seconds_binary}",
                    f"--timeout_seconds_gem5 {self.timeout_seconds_gem5}"]
        if self.use_logical_cpus:
            command.append("--use_logical_cpus")
        if self.threaded:
            command.append("--threaded")
        if self.exit_early_on_fail:
            command.append("--exit_early_on_fail")
        return command
    
    def _find_open_port(self):
        """
        https://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python
        """
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("",0))
        s.listen(1)
        port = s.getsockname()[1]
        s.close()
        return port
    
    def sanity_check_port(self):
        import socket
        
        def _is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
            
        if _is_port_in_use(self.port):
            logging.warning(f"Port {self.port} is already in use, finding another port")
            self.port = self._find_open_port()
            logging.warning(f"New port: {self.port}")
    
    def run_without_container(self):
        
        self.sanity_check_port()
        arch_arg = self._get_arch_arg()
        
        command = self.build_gem5_command(arch_arg)
        command = shlex.split(" ".join(command))
        # command = self.build_gem5_command()
        def run_command():
            subprocess.run(command)
        self.child_process = multiprocessing.Process(target=run_command)
        self.child_process.start()
        
        
    def stream_docker_logs(self): 
        container = self.client.containers.get(self.container.name)
        for line in container.logs(stream=True, follow=True):
            print(line.decode('utf-8').strip())
    
    def start_stream_thread(self):
        # threading.Thread(target=stream_docker_logs
        self.stream_thread = threading.Thread(target=self.stream_docker_logs)
        self.stream_thread.start()
    
    def stop_stream_thread(self):
        pass
        # assert self.stream_thread is not None
        # self.stream_thread.join()
        # self.stream_thread = None
        
        
    def teardown(self, remove_container=True):
        if self.do_run_without_container: 
            self.child_process.terminate()
            self.child_process.join()
            logging.info("Child process terminated")
        else: 
            self.container.stop()
            if remove_container:
                self.container.remove()
            self.client.close()

    def submit_single_submission(self, 
                             code: str, 
                             testcases: List[str], 
                             problem_id: str, 
                             timing_env: str, 
                             override_flags: str = None):
        
        print(f"Submitting single submission to port {self.port}")
        
        req = requests.get(f"http://localhost:{self.port}/gem5/single_submission", 
                        json={"code": code, 
                              "testcases": testcases, 
                              "problem_id": problem_id, 
                              "timing_env": timing_env, 
                              "override_flags": override_flags, 
                              "api_key": self.api_key})
        # return req.json()
        return parse_submission_result(req.json())

    def submit_single_submission_pair(self, code_v0: str, 
                                    code_v1: str, 
                                    testcases: List[str], 
                                    problem_id: str, 
                                    timing_env: str, 
                                    override_flags: str = None):
        
        print(f"Submitting single submission pair to port {self.port}")
        
        req = requests.get(f"http://localhost:{self.port}/gem5/single_submission_pair", 
                        json={"code_v0": code_v0, 
                            "code_v1": code_v1, 
                            "testcases": testcases, 
                            "problem_id": problem_id, 
                            "timing_env": timing_env, 
                            "override_flags": override_flags, 
                            "api_key": self.api_key})
        # return req.json()
        return parse_submission_result(req.json())


    def _get_multiple_single_submissions(self, submissions: List[Dict[str, str]], 
                                        timing_env: str):
        
        # self.start_stream_thread()
        req = requests.get(f"http://localhost:{self.port}/gem5/multiple_single_submissions", 
                            json={"submissions": submissions, 
                                "timing_env": timing_env, 
                                "api_key": self.api_key})

        # self.stop_stream_thread()
        return parse_submission_result(req.json())

    def submit_multiple_single_submissions(self, code_list: List[str],
                                            testcases_list: List[List[str]],
                                            problem_id_list: List[str],
                                            timing_env: str,
                                            override_flags_list: List[str] = None):
        print(f"Submitting multiple single submissions to port {self.port}")
        if override_flags_list is None:
            override_flags_list = [None] * len(code_list)
        submissions = [{"code": code,
                        "testcases": testcases,
                        "problem_id": problem_id,
                        "override_flags": override_flags} 
                        for code, testcases, problem_id, override_flags 
                        in zip(code_list, testcases_list, problem_id_list, override_flags_list)]
        return self._get_multiple_single_submissions(submissions, timing_env)


    def _get_multiple_dual_submissions(self, submissions_v0: List[Dict[str, str]], 
                                        submissions_v1: List[Dict[str, str]], 
                                        timing_env: str):
        req = requests.get(f"http://localhost:{self.port}/gem5/multiple_submissions_pairs", 
                            json={"submissions_v0": submissions_v0, 
                                "submissions_v1": submissions_v1, 
                                "timing_env": timing_env, 
                                "api_key": self.api_key})
        # return req.json()
        return parse_submission_result(req.json())

    def submit_multiple_dual_submissions(self, code_list_v0: List[str],
                                            code_list_v1: List[str],
                                            testcases_list: List[List[str]],
                                            problem_id_list: List[str],
                                            timing_env: str,
                                            override_flags_list_v0: List[str] = None,
                                            override_flags_list_v1: List[str] = None):
        
        print(f"Submitting multiple dual submissions to port {self.port}")
        if override_flags_list_v0 is None:
            override_flags_list_v0 = [None] * len(code_list_v0)
        if override_flags_list_v1 is None:
            override_flags_list_v1 = [None] * len(code_list_v1)
        submissions_v0 = [{"code": code,
                        "testcases": testcases,
                        "problem_id": problem_id,
                        "override_flags": override_flags} 
                        for code, testcases, problem_id, override_flags 
                        in zip(code_list_v0, testcases_list, problem_id_list, override_flags_list_v0)]
        submissions_v1 = [{"code": code,
                        "testcases": testcases,
                        "problem_id": problem_id,
                        "override_flags": override_flags} 
                        for code, testcases, problem_id, override_flags 
                        in zip(code_list_v1, testcases_list, problem_id_list, override_flags_list_v1)]
        return self._get_multiple_dual_submissions(submissions_v0, submissions_v1, timing_env)
    
    def test_connection(self):
        try: 
            req = requests.get(f"http://localhost:{self.port}/gem5/ping", 
                                json={"api_key": self.api_key})
        
            if req.status_code == 200:
                return True
            else:
                return False
        except:
            return False
        
    def wait_for_connection(self, timeout=5):
        print(f"Waiting for connection to port {self.port}")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.test_connection():
                print(f"Connection to port {self.port} established")
                return True
            time.sleep(0.25)
        print(f"Connection to port {self.port} timed out")
        return False
        
        
        
        
        