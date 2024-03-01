## PIE Experiments

The following directory containe the outputs from our experiments generally in the following format 

- `aggregated_test_results.csv`: contains final statistics including those found in the paper as well as additional statistics, such as %Opt reported in the paper at higher thresholds for speedup like 1.5x, 2.0x and so on. 
- `test_results.jsonl`: contains outputs and additional metrics after benchmarking with `gem5`
- `melted_test_results.jsonl`: this is an intermediate file containing test_results.jsonl melted into a long format for easier analysis.
- `additional_test_results.jsonl`: is an intermediate file which we did not use and may or may not contain additional `gem5` benchmarking information. 
- `raw_test_results.jsonl`: is an intermediate file for test_results.jsonl without benchmarking info. This usually will exist, but in some cases it won't. test_results.jsonl and melted_test_results.jsonl are derived from this file and will also contain any information in this file in addition to more. 