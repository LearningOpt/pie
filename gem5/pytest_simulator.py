import pytest
from gem5 import simulator
from gem5.simulator import PieEnvironment, PieSingleResult, PiePairResult, make
import numpy as np
from collections import defaultdict
from pprint import pprint

API_KEY="cdZ5TynkL5D7gCTFvzJT4YKu05aozTLp4GgIcK5"

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

example_2_code = """
#include<iostream>
#include<bits/stdc++.h>
typedef long long ll;
typedef unsigned int ui;
#define infin (ll)(998244353)
using namespace std;
int main()
{
   int n,k;
   cin>>n>>k;
   int l,r;
   vector<ll> dp(n+1,0); //0 to n
   vector <pair<int,int>>v;
   for(int j=0;j<k;j++)
   {
      cin>>l>>r;
      v.push_back({l,r});
   }
   dp[0]=1;;
   dp[1]=1;
   sort(v.begin(),v.end());
   auto z=v.begin();
   if ((*z).first==1)
      dp[2]=1;
   else
     dp[2]=0;
   for(int i=3;i<=n;i++)
   {
      dp[i]=dp[i-1];
      for (auto x:v)
      {
         if (i>x.first)
            dp[i]+=dp[i-x.first];
         else
            break;
         if (i-1>x.second)
            {
               dp[i]-=dp[i-1-x.second];
               if (dp[i]<0)
                  dp[i]+=infin;
            }
      }
      dp[i]=(dp[i]) % infin;
   }
   cout<<dp[n]<<endl;
}
"""

example_2_problem_id = "p02549"

@pytest.fixture(scope='session', autouse=True)
def get_pie_env():
    env = simulator.make(api_key=API_KEY)
    yield env
    env.teardown()

class TestFrontEnd:
   def test_multiple_dual_submissions(self, get_pie_env):

      # use example_1_code 2 times; the only difference is that the keys are gem5_v0 and gem5_v1 and binary_v0 and binary_v1
      tc2hyperfine_v0 = defaultdict(list)
      tc2hyperfine_v1 = defaultdict(list)

      code_v0_list = [example_1_code] * 3
      code_v1_list = [example_2_code] * 3
      testcases_list = [[0, 1]] * 3
      problem_id_list = [example_1_problem_id] * 3
      override_flags_list = [""] * 3

      env = get_pie_env
      # import pdb; pdb.set_trace()
      results = env.submit_multiple_dual_submissions(code_list_v0=code_v0_list,
                                                      code_list_v1=code_v1_list,
                                                      testcases_list=testcases_list,
                                                      problem_id_list=problem_id_list,
                                                      timing_env="both",
                                                      override_flags_list_v0=override_flags_list,
                                                      override_flags_list_v1=override_flags_list)

      for result in results:

         assert result.compilation_v0 == True
         assert result.compilation_v1 == True

         assert result.mean_acc_v0 > 0.95
         assert result.mean_acc_v1 > 0.95

         pprint(result.tc2time_v0)
         pprint(result.tc2time_v1)

         print(
               f"result.tc2time_v0[0] = {result.tc2time_v0[0]} should be 0.001035073468")
         print(
               f"result.tc2time_v0[1] = {result.tc2time_v0[1]} should be 0.001039205596")
         print(
               f"result.tc2time_v1[0] = {result.tc2time_v1[0]} should be 0.001026564396")
         print(
               f"result.tc2time_v1[1] = {result.tc2time_v1[1]} should be 0.001029346032")

         assert result.tc2time_v0[0] == 0.001035073468
         assert result.tc2time_v0[1] == 0.001039205596

         assert result.tc2time_v1[0] == 0.001026564396
         assert result.tc2time_v1[1] == 0.001029346032

         hyperfine_v0_tc2stats = result.tc2stats_binary_v0
         hyperfine_v1_tc2stats = result.tc2stats_binary_v1

         for tc, time in hyperfine_v0_tc2stats.items():
               tc2hyperfine_v0[tc].append(np.array(time))
         for tc, time in hyperfine_v1_tc2stats.items():
               tc2hyperfine_v1[tc].append(np.array(time))

      for tc, times_v0 in tc2hyperfine_v0.items():
         mean_times_v0 = []
         for time_list in times_v0:
            mean_times_v0.append(np.mean(time_list))
         mean_times_v1 = []
         for time_list in tc2hyperfine_v1[tc]:
            mean_times_v1.append(np.mean(time_list))
         # consistency check
         assert (np.std(mean_times_v0) / np.mean(mean_times_v0)
                  ) < 0.05, f"std/mean = {np.std(mean_times_v0) / np.mean(mean_times_v0)} for tc {tc} with mean times {mean_times_v0}"
         assert (np.std(mean_times_v1) / np.mean(mean_times_v1)
                  ) < 0.05, f"std/mean = {np.std(mean_times_v1) / np.mean(mean_times_v1)} for tc {tc} with mean times {mean_times_v1}"
         # performance check
         assert (np.mean(mean_times_v0) / np.mean(mean_times_v1)
                  ) > .95, f"mean_times_v0 {mean_times_v0} mean_times_v1 {mean_times_v1}"
         print(
               f"std/mean v0 tc {tc}= {np.std(mean_times_v0) / np.mean(mean_times_v0)} for tc {tc} with mean times {mean_times_v0} ")
         print(
               f"std/mean v1 tc {tc}= {np.std(mean_times_v1) / np.mean(mean_times_v1)} for tc {tc} with mean times {mean_times_v1} ")
         print(
               f"mean_times_v0 {mean_times_v0} mean_times_v1 {mean_times_v1} for tc {tc}, with speedup {np.mean(mean_times_v0) / np.mean(mean_times_v1)}")

      assert len(tc2hyperfine_v0) == 2
      assert len(tc2hyperfine_v1) == 2


   def test_single_submission(self, get_pie_env):
      env = get_pie_env
      tc2hyperfine = defaultdict(list)
      for _ in range(2): 
         result = env.submit_single_submission(code=example_1_code,
                                                testcases=[0,1],
                                                problem_id=example_1_problem_id,
                                                timing_env="both")

         assert result.compilation == True 
         assert result.tc2success[0] == True 
         assert result.tc2success[1] == True 
         assert result.tc2time[0] == 0.001035073468
         assert result.tc2time[1] == 0.001039205596
         assert result.mean_acc > 0.95
         
         hyperfine_result = result.tc2stats_binary
         
         for tc, results in hyperfine_result.items():
            tc2hyperfine[tc].append(np.array(results))

      for tc, times in tc2hyperfine.items():
         mean_times = []
         for time_list in times:
               mean_times.append(np.mean(time_list))
         assert (np.std(mean_times) / np.mean(mean_times)) < 0.05, f"std/mean = {np.std(mean_times) / np.mean(mean_times)} for tc {tc} with mean times {mean_times}"
      assert len(tc2hyperfine) == 2
      


   def test_dual_submission_diff_code(self, get_pie_env):
      env = get_pie_env
      tc2hyperfine_v0 = defaultdict(list)
      tc2hyperfine_v1 = defaultdict(list)
      for _ in range(2): 
         result = env.submit_single_submission_pair(code_v0=example_1_code, 
                                                   code_v1=example_2_code,
                                                   testcases=[0,1],
                                                   problem_id=example_1_problem_id,
                                                   timing_env="both")
         
         
         assert result.compilation_v0 == True
         assert result.compilation_v1 == True
         
         assert result.mean_acc_v0 > 0.95
         assert result.mean_acc_v1 > 0.95
         
         pprint(result.tc2time_v0)
         pprint(result.tc2time_v1)
         
         print(f"result.tc2time_v0[0] = {result.tc2time_v0[0]} should be 0.001035073468")
         print(f"result.tc2time_v0[1] = {result.tc2time_v0[1]} should be 0.001039205596")
         print(f"result.tc2time_v1[0] = {result.tc2time_v1[0]} should be 0.001026564396")
         print(f"result.tc2time_v1[1] = {result.tc2time_v1[1]} should be 0.001029346032")
         
         assert result.tc2time_v0[0] == 0.001035073468
         assert result.tc2time_v0[1] == 0.001039205596
         
         assert result.tc2time_v1[0] == 0.001026564396
         assert result.tc2time_v1[1] == 0.001029346032
         
         hyperfine_v0_tc2stats = result.tc2stats_binary_v0
         hyperfine_v1_tc2stats = result.tc2stats_binary_v1

         for tc, time in hyperfine_v0_tc2stats.items():
               tc2hyperfine_v0[tc].append(np.array(time))
         for tc, time in hyperfine_v1_tc2stats.items():
               tc2hyperfine_v1[tc].append(np.array(time))
               
      for tc, times_v0 in tc2hyperfine_v0.items():
         mean_times_v0 = []
         for time_list in times_v0 :	
               mean_times_v0.append(np.mean(time_list))
         mean_times_v1 = []
         for time_list in tc2hyperfine_v1[tc] :	
               mean_times_v1.append(np.mean(time_list))
         # consistency check
         assert (np.std(mean_times_v0) / np.mean(mean_times_v0)) < 0.05, f"std/mean = {np.std(mean_times_v0) / np.mean(mean_times_v0)} for tc {tc} with mean times {mean_times_v0}"
         assert (np.std(mean_times_v1) / np.mean(mean_times_v1)) < 0.05, f"std/mean = {np.std(mean_times_v1) / np.mean(mean_times_v1)} for tc {tc} with mean times {mean_times_v1}"
         # performance check
         assert (np.mean(mean_times_v0) / np.mean(mean_times_v1)) > .95, f"mean_times_v0 {mean_times_v0} mean_times_v1 {mean_times_v1}"
         print(f"std/mean v0 tc {tc}= {np.std(mean_times_v0) / np.mean(mean_times_v0)} for tc {tc} with mean times {mean_times_v0} ")
         print(f"std/mean v1 tc {tc}= {np.std(mean_times_v1) / np.mean(mean_times_v1)} for tc {tc} with mean times {mean_times_v1} ")
         print(f"mean_times_v0 {mean_times_v0} mean_times_v1 {mean_times_v1} for tc {tc}, with speedup {np.mean(mean_times_v0) / np.mean(mean_times_v1)}")

      assert len(tc2hyperfine_v0) == 2
      assert len(tc2hyperfine_v1) == 2
   
   
   def test_dual_submission_same_code(self, get_pie_env):
      env = get_pie_env
      tc2hyperfine_v0 = defaultdict(list)
      tc2hyperfine_v1 = defaultdict(list)
      for _ in range(2): 
         result = env.submit_single_submission_pair(code_v0=example_1_code, 
                                                   code_v1=example_1_code,
                                                   testcases=[0,1],
                                                   problem_id=example_1_problem_id,
                                                   timing_env="both")
         
         
         assert result.compilation_v0 == True
         assert result.compilation_v1 == True
         
         assert result.mean_acc_v0 > 0.95
         assert result.mean_acc_v1 > 0.95
         
         pprint(result.tc2time_v0)
         pprint(result.tc2time_v1)
         
         print(f"result.tc2time_v0[0] = {result.tc2time_v0[0]} should be 0.001035073468")
         print(f"result.tc2time_v0[1] = {result.tc2time_v0[1]} should be 0.001039205596")
         print(f"result.tc2time_v1[0] = {result.tc2time_v1[0]} should be 0.001035073468")
         print(f"result.tc2time_v1[1] = {result.tc2time_v1[1]} should be 0.001039205596")
         
         assert result.tc2time_v0[0] == 0.001035073468
         assert result.tc2time_v0[1] == 0.001039205596
         
         assert result.tc2time_v1[0] == 0.001035073468
         assert result.tc2time_v1[1] == 0.001039205596
         
         hyperfine_v0_tc2stats = result.tc2stats_binary_v0
         hyperfine_v1_tc2stats = result.tc2stats_binary_v1

         for tc, time in hyperfine_v0_tc2stats.items():
               tc2hyperfine_v0[tc].append(np.array(time))
         for tc, time in hyperfine_v1_tc2stats.items():
               tc2hyperfine_v1[tc].append(np.array(time))
               
      for tc, times_v0 in tc2hyperfine_v0.items():
         times_v1 = tc2hyperfine_v1[tc]
         mean_times = []
         for time_list in times_v0 + times_v1:	
            mean_times.append(np.mean(time_list))
         assert (np.std(mean_times) / np.mean(mean_times)) < 0.05, f"std/mean = {np.std(mean_times) / np.mean(mean_times)} for tc {tc} with mean times {mean_times}"
         print(f"std/mean = {np.std(mean_times) / np.mean(mean_times)} for tc {tc} with mean times {mean_times} ")
      assert len(tc2hyperfine_v0) == 2


    
   def test_multiple_single_submissions(self, get_pie_env):
      
      
         code_list = [example_1_code, example_2_code] * 3
         testcases_list = [[0, 1], [0, 1]] * 3
         problem_id_list = [example_1_problem_id, example_2_problem_id] * 3
         override_flags_list = ["", ""] * 3
         
         env = get_pie_env

         results = env.submit_multiple_single_submissions(code_list=code_list,
                                                            testcases_list=testcases_list,
                                                            problem_id_list=problem_id_list,
                                                            override_flags_list=override_flags_list,
                                                            timing_env="both")

         tc2hyperfine_v0 = defaultdict(list)
         tc2hyperfine_v1 = defaultdict(list)

         for i, result in enumerate(results):
            assert result.compilation == True 
            assert result.tc2success[0] == True 
            assert result.tc2success[1] == True 
            
            hyperfine_result = result.tc2stats_binary

            if (i % 2) == 0: 
               assert result.tc2time[0] == 0.001035073468
               assert result.tc2time[1] == 0.001039205596
               tc2hyperfine = tc2hyperfine_v0
            else: 
               assert result.tc2time[0] == 0.001026564396
               assert result.tc2time[1] == 0.001029346032
               tc2hyperfine = tc2hyperfine_v1
               
            for tc, results in hyperfine_result.items():
               tc2hyperfine[tc].append(np.array(results))

         for tc, times_v0 in tc2hyperfine_v0.items():
            mean_times_v0 = []
            for time_list in times_v0 :	
               mean_times_v0.append(np.mean(time_list))
            mean_times_v1 = []
            for time_list in tc2hyperfine_v1[tc] :	
               mean_times_v1.append(np.mean(time_list))
      
            print(f"std/mean v0 tc {tc}= {np.std(mean_times_v0) / np.mean(mean_times_v0)} for tc {tc} with mean times {mean_times_v0} ")
            print(f"std/mean v1 tc {tc}= {np.std(mean_times_v1) / np.mean(mean_times_v1)} for tc {tc} with mean times {mean_times_v1} ")
            print(f"mean_times_v0 {mean_times_v0} mean_times_v1 {mean_times_v1} for tc {tc}, with speedup {np.mean(mean_times_v0) / np.mean(mean_times_v1)}")
      
            # consistency check
            assert (np.std(mean_times_v0) / np.mean(mean_times_v0)) < 0.05, f"std/mean = {np.std(mean_times_v0) / np.mean(mean_times_v0)} for tc {tc} with mean times {mean_times_v0}"
            assert (np.std(mean_times_v1) / np.mean(mean_times_v1)) < 0.05, f"std/mean = {np.std(mean_times_v1) / np.mean(mean_times_v1)} for tc {tc} with mean times {mean_times_v1}"
            # performance check
            assert (np.mean(mean_times_v0) / np.mean(mean_times_v1)) > .95, f"mean_times_v0 {mean_times_v0} mean_times_v1 {mean_times_v1}"
            
      
         assert len(tc2hyperfine_v0) == 2
         assert len(tc2hyperfine_v1) == 2