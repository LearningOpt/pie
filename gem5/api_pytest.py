import benchmarking
import tempfile 
import subprocess 
import os 
import glob
import numpy as np
from tqdm import tqdm
from collections import defaultdict

count_to_10_cpp = """
#include <iostream>
using namespace std;

int main() {
    for (int i = 0; i < 10; i++) {
        cout << i << endl;
    }
    return 0;
}
""" 

mult_in_by_2_cpp = """
#include <iostream>
using namespace std;

int main() {
    int x;
    cin >> x;
    cout << x * 2 << endl;
    return 0;
}
"""

example_1_code = """
#include <bits/stdc++.h>
#define REP(i, n) for (int i = 0; i < (n); i++)
using namespace std;
const int MOD = 998244353;

int main() {
	cin.tie(0)->sync_with_stdio(false);

	int n, k; cin >> n >> k;
	vector<int> l(k), r(k);
	REP(i, k) cin >> l[i] >> r[i];
	REP(i, k) r[i]++;

	vector<int> dp(n + 1, 0);
	dp[0] = 1;
	dp[1] = -1;
	REP(i, n) {
		if (i > 0)
			dp[i] = (dp[i] + dp[i - 1]) % MOD;
		REP(j, k) {
			if (i + l[j] < n)
				dp[i + l[j]] = (dp[i + l[j]] + dp[i]) % MOD;
			if (i + r[j] < n)
				dp[i + r[j]] = (((dp[i + r[j]] - dp[i]) % MOD) + MOD) % MOD;
		}
	}
	cout << dp[n - 1] << endl;
	return 0;
}
"""
example_1_problem_id = "p02549"

example_hello_world_code = """
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""

# def exec_bin_for_acc(bin_path, in_path, ground_truth_output, timeout):
#     logging.info(f'executing {bin_path}, with input {in_path}')
#     with open(in_path, 'r') as fh:
#         p = subprocess.run([bin_path], capture_output=True, timeout=timeout, stdin=fh, text=True)
#     if p.returncode != 0:
#         raise Exception(f"Error executing code: {bin_path}, return code: {p.returncode}, stderr: {p.stderr.decode('utf-8')}")
#     else: 
#         return get_accuracy(p.stdout, ground_truth_output)


class TestBenchmarking: 
    def test_compile(self): 
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "basic.cpp")
            with open(code_path, "w") as f:
                f.write(count_to_10_cpp)
            output_path = benchmarking.compile_cpp_code(code_path)
            p = subprocess.run([output_path], capture_output=True, text=True)
            assert p.returncode == 0
            assert p.stdout.strip() == "\n".join([str(i) for i in range(10)])
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
    def test_exec_bin(self): 
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "basic.cpp")
            with open(code_path, "w") as f:
                f.write(count_to_10_cpp)
            output_path = benchmarking.compile_cpp_code(code_path)
            rc, stdout, stderr = benchmarking.exec_bin(output_path, None, None)
            assert rc == 0
            assert stdout.strip() == "\n".join([str(i) for i in range(10)])
            assert stderr == ""
            
    def test_exec_bin_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "basic.cpp")
            input_path = os.path.join(tmpdir, "input.txt")
            with open(code_path, "w") as f:
                f.write(mult_in_by_2_cpp)
            with open(input_path, "w") as f:
                f.write("2")
            output_path = benchmarking.compile_cpp_code(code_path)
            rc, stdout, stderr = benchmarking.exec_bin(output_path, input_path, None)
            assert rc == 0
            assert stdout.strip() == "4"
            assert stderr == ""
            
    def test_exec_bin_for_acc(self):    
         with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "basic.cpp")
            input_path = os.path.join(tmpdir, "input.txt")
            with open(code_path, "w") as f:
                f.write(mult_in_by_2_cpp)
            with open(input_path, "w") as f:
                f.write("2")
            output_path = benchmarking.compile_cpp_code(code_path)
            acc_correct = benchmarking.exec_bin_for_acc(output_path, input_path, "4", None)
            acc_incorrect = benchmarking.exec_bin_for_acc(output_path, input_path, "5", None)
            assert acc_correct == 1
            assert acc_incorrect == 0
            
    def test_compile_and_check_outputs(self): 
        with tempfile.TemporaryDirectory() as tempdir: 
            code_path = os.path.join(tempdir, "basic.cpp")
            with open(code_path, "w") as fh: 
                fh.write(example_1_code)
            bin_path, accs = benchmarking.compile_and_check_outputs(
                code_path=code_path, 
                problem_id=example_1_problem_id, 
                testcases_dir="/home/pie-perf/data/codenet/merged_test_cases/"
            )
            print(f"bin_path: {bin_path}")
            assert os.path.exists(bin_path)
            assert os.path.getsize(bin_path) > 0
        assert np.mean(list(accs.values())) == 1.0
        assert np.std(list(accs.values())) == 0.0
        n_testcases = len(glob.glob(os.path.join("/home/pie-perf/data/codenet/merged_test_cases/", example_1_problem_id, "input.*.txt")))
        assert len(accs) == n_testcases
        
    def test_exec_gem5(self):
        sim_seconds = []
        sim_seconds_precise = []
        for _ in tqdm(range(5)):
            with tempfile.TemporaryDirectory() as tmpdir:
                code_path = os.path.join(tmpdir, "basic.cpp")
                with open(code_path, "w") as f:
                    f.write(example_hello_world_code)
                output_path = benchmarking.compile_cpp_code(code_path, cflags="--std=c++17 -O3")
                rc, stdout, stderr = benchmarking.exec_gem5(
                    gem5_dir="/home/gem5/build/X86/", 
                    gem5_script_path="/home/gem5-skylake-config/gem5-configs/run-se.py", 
                    cpu_type="Verbatim",
                    bin_path=output_path,
                    in_path=None,
                    stats_out_path=os.path.join(tmpdir, "stats.txt"),
                    timeout=60, 
                    cpu_number=0)
                
                assert rc == 0
                stats = benchmarking.parse_stats_txt(os.path.join(tmpdir, "stats.txt"))
                sim_seconds.append(stats["sim_seconds"])
                sim_seconds_precise.append(stats["sim_seconds_precise"])
        print(f"sim_seconds: {sim_seconds}")
        print(f"sim_seconds_precise: {sim_seconds_precise}")
        assert np.isclose(np.mean(sim_seconds), 0.001004, atol=1e-5)
        assert np.isclose(np.mean(sim_seconds_precise), 0.001004, atol=1e-5)
        assert all(sim_seconds_precise[i] == 0.001004121118 for i in range(len(sim_seconds_precise)))

    def test_run_gem5(self): 
        sim_seconds_0 = []
        sim_seconds_1 = []
        for _ in tqdm(range(2)): 
            with tempfile.TemporaryDirectory() as tmpdir:
                code_path = os.path.join(tmpdir, "code.cpp")
                with open(code_path, "w") as f:
                    f.write(example_1_code)
                bin_path = benchmarking.compile_cpp_code(code_path)
                tc_2_results = benchmarking.run_gem5(
                    gem5_dir="/home/gem5/build/X86/", 
                    gem5_script_path="/home/gem5-skylake-config/gem5-configs/run-se.py", 
                    cpu_type="Verbatim",
                    bin_path=bin_path, 
                    problem_id=example_1_problem_id, 
                    testcases_dir="/home/pie-perf/data/codenet/merged_test_cases/", 
                    testcases=[0,1], 
                    timeout=30, 
                    cpu_number=0
                )
                assert tc_2_results[0]["success"] == True 
                assert tc_2_results[1]["success"] == True 
                assert len(tc_2_results) == 2 
                sim_seconds_0.append(tc_2_results[0]["stats"]["sim_seconds_precise"])
                sim_seconds_1.append(tc_2_results[1]["stats"]["sim_seconds_precise"])
        print(f"sim_seconds for tc 0 {sim_seconds_0}")
        print(f"sim_seconds for tc 1 {sim_seconds_1}")
        assert sim_seconds_0[0] == sim_seconds_0[1] == 0.001035073468
        assert sim_seconds_1[0] == sim_seconds_1[1] == 0.001039205596
    

    def test_run_hyperfine(self):
        tc2times = defaultdict(list)
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmpdir:
                    code_path = os.path.join(tmpdir, "code.cpp")
                    with open(code_path, "w") as f:
                        f.write(example_1_code)
                    code2results, output = benchmarking.run_hyperfine(
                        code_paths=[code_path],
                        problem_ids=[example_1_problem_id],
                        path_to_testcases="/home/pie-perf/data/codenet/merged_test_cases/",
                        json_out_path=os.path.join(tmpdir, "results.json"),
                        test_cases_list=[[i for i in range(10)]], 
                        min_runs_per_test_case=10, 
                        max_runs_per_test_case=500, 
                        strict_runs_per_test_case=False,
                        warmup_runs_per_test_case=5,
                        cpu_number=0,
                        do_sanity_check=True)
                    for tc, results in code2results[code_path].items():
                        tc2times[tc].append(np.array(results["times"]))
        for tc, times in tc2times.items():
            mean_times = []
            for time_list in times:
                mean_times.append(np.mean(time_list))
            assert (np.std(mean_times) / np.mean(mean_times)) < 0.05, f"std/mean = {np.std(mean_times) / np.mean(mean_times)} for tc {tc} with mean times {mean_times}"
            print(f"std/mean = {np.std(mean_times) / np.mean(mean_times)} for tc {tc} with mean times {mean_times} ")
        assert len(tc2times) == 10
        
    def test_run_hyperfine_strict(self):
        tc2times = defaultdict(list)
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmpdir:
                    code_path = os.path.join(tmpdir, "code.cpp")
                    with open(code_path, "w") as f:
                        f.write(example_1_code)
                    code2results, output = benchmarking.run_hyperfine(
                        code_paths=[code_path],
                        problem_ids=[example_1_problem_id],
                        path_to_testcases="/home/pie-perf/data/codenet/merged_test_cases/",
                        json_out_path=os.path.join(tmpdir, "results.json"),
                        test_cases_list=None, 
                        min_runs_per_test_case=100, 
                        max_runs_per_test_case=None, 
                        strict_runs_per_test_case=True,
                        warmup_runs_per_test_case=5,
                        cpu_number=0,
                        do_sanity_check=True)
                    for tc, results in code2results[code_path].items():
                        tc2times[tc].append(np.array(results["times"]))
        for tc, times in tc2times.items():
            assert len(times) == 2
            mean_times = []
            for time_list in times:
                assert len(time_list) == 100
                mean_times.append(np.mean(time_list))
            assert (np.std(mean_times) / np.mean(mean_times)) < 0.05, f"std/mean = {np.std(mean_times) / np.mean(mean_times)} for tc {tc} with mean times {mean_times}"
            print(f"std/mean = {np.std(mean_times) / np.mean(mean_times)} for tc {tc} with mean times {mean_times} ")
        assert len(tc2times) == len(glob.glob(f"/home/pie-perf/data/codenet/merged_test_cases/{example_1_problem_id}/input*"))
            
            
                

            
            
    
    
        