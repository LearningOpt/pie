# from src.codenet_eval.run_eval import (read_ground_truths, read_inputs_and_prepare)
# from src.codenet_eval.evalconfig import EvaluationConfig
import tarfile                   
import shutil        
import tempfile
import logging 
import pandas as pd
import json
import os 
import pdb
import argparse
from gem5.simulator import PieEnvironment
from gem5 import simulator
import traceback
import pdb
import threading
from tqdm import tqdm
import re
from typing import Optional, Any
import yaml
from dataclasses import dataclass, field
import ast

logging.basicConfig(level=logging.INFO)

import signal
import time

KEY_COLS = ["n_tests", 
            "problem_id", 
            "tests"
            "src_id", 
            "tgt_id", 
            "fastest_runtime", "fastest_accuracy"]


def get_key_columns(df, cfg):
    ## in key columns or if 
    ## *_test_compilation, *_test_accuracy, *_test_agg_runtime, *_tc2time
    key_cols = [c for c in df.columns if c in KEY_COLS or c.endswith("_compilation") or c.endswith("_accuracy") or c.endswith("_runtime") or c.endswith("_tc2time")]
    key_cols += [c for c in df.columns if cfg.model_generated_potentially_faster_code_col in c] + [cfg.slow_code_col, cfg.reference_code_col]
    key_cols = list(set(key_cols))
    return df[key_cols]

def _fix_value(x: Any) -> Any:
    ## if starts with '[' and ends with ']', as a string, then convert to list
    if isinstance(x, str) and len(x) > 1 and x[0] == '[' and x[-1] == ']':
        x = ast.literal_eval(x)
    return x

def fix_df_columns(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: _fix_value(x))
    return df
    


def unmelt_results(results_df, cfg, remove_extra_cols=False):
        unmelted_data = []
        for src_id, group in results_df.groupby("src_id"):
            src_code_row = group[group["code_type"] == "src_code"].iloc[0]
            new_row = src_code_row.to_dict()
            for index, row in group.iterrows():
                new_row["src_id"] = src_id
                new_row[f'{row["code_type"]}_compilation'] = row["compilation"]
                new_row[f'{row["code_type"]}'] = row["code"]
                if row["code_type"].startswith(cfg.model_generated_potentially_faster_code_col) or cfg.redo_src_tgt:
                    new_row[f'{row["code_type"]}_accuracy'] = row["accuracy"]
                    new_row[f'{row["code_type"]}_agg_runtime'] = row["agg_runtime"]
                    new_row[f'{row["code_type"]}_tc2time'] = row["tc2time"]
            unmelted_data.append(new_row)
        ## clean up the column names
        unmelted_df = pd.DataFrame(unmelted_data)
        if remove_extra_cols:
            unmelted_df = get_key_columns(unmelted_df, cfg)
        
        # unmelted_df = rename_columns(unmelted_df)
        
        return unmelted_df
        
def report_results(df, cfg, orig_df): 
        ## all columns will be cfg.model_generated_potentially_faster_code_col_*
        ## for these, consider only use those that are not None, above threshold_accuracy, and have the fastest_runtime
        ## for those, keep the runtime, but if the accuracy is below threshold_accuracy, set the runtime to float("inf")
        
        ## then consider only max_generations_to_report
        
        ## in 1, 2, 4... (powers of 2 up until len(runtimes)), report the best runtime
        ## as runtime_best@1, runtime_best@2, runtime_best@4, etc. accuracy_best@1, accuracy_best@2, accuracy_best@4, etc.
        ## while also reporting speedup_best@1, speedup_best@2, speedup_best@4, etc. where speedup = runtime_src / runtime_best@n 
        
        
        ## then aggregate 
        ### 1. for each 1, 2.. (powers of 2 up until len(runtimes)), report mean_accuracy@n, mean_speedup@n where we also take speedup = min(1.0, runtime_src / runtime_best@n)
        ### 2. for each 1, 2.. (powers of 2 up until len(runtimes)), report the % of programs where the speedup is >= 1.10, 1.25, 1.50, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0
    
        # merged[f"{cfg.model_generated_potentially_faster_code_col}_{i}"] = merged[cfg.model_generated_potentially_faster_code_col].apply(lambda x: x[i] if i < len(x) else None)
        import pdb
        # pdb.set_trace()
        # print("columns before report_results")
        # print(df.columns)
        
        
        # num_generated_cols = len([c for c in df.columns if re.match(f"{cfg.model_generated_potentially_faster_code_col}_[0-9]+", c) or c == cfg.model_generated_potentially_faster_code_col])
        num_generated_cols = cfg.num_generated_cols
        assert num_generated_cols is not None, f"num_generated_cols is None, it should have been set in read_inputs_and_prepare_v2"
        
        import pandas as pd
        import numpy as np

        # Assuming orig_df and df are already defined, and cfg and num_generated_cols are given

        # Step 1: Find rows in orig_df that are not in df
        # do this with src_code not src_id 
        print(f"length of orig_df {len(orig_df)} vs length of results_df {len(df)}")
        orig_df["src_tgt_code"] = orig_df[cfg.slow_code_col] + orig_df[cfg.reference_code_col]
        df["src_tgt_code"] = df[cfg.slow_code_col] + df[cfg.reference_code_col]
        # drop duplicates from both 
        df = df.drop_duplicates(subset=["src_tgt_code"])
        orig_df = orig_df.drop_duplicates(subset=["src_tgt_code"])
        unique_rows = orig_df[~orig_df['src_tgt_code'].isin(df['src_tgt_code'])]
        assert len(unique_rows) == (len(orig_df) - len(df)), f"len(unique_rows) {len(unique_rows)} == len(orig_df) - len(df) {len(orig_df) - len(df)}"

        # Step 2: Create additional columns for the unique rows and set default values
        for j in range(num_generated_cols + 1):  # Adding 1 to include the case when j == num_generated_cols
            colname = f"{cfg.model_generated_potentially_faster_code_col}_{j}" if num_generated_cols > 0 else cfg.model_generated_potentially_faster_code_col
            unique_rows[f"{colname}_agg_runtime"] = float("inf")  # Setting runtime to inf
            unique_rows[f"{colname}_accuracy"] = 0  # Setting accuracy to 0
            unique_rows[f"{colname}_tc2time"] = [{} for _ in range(len(unique_rows))]  # Setting tc2time to {}
        # drop unique rows columns that are not in df
        unique_rows = unique_rows[[c for c in unique_rows.columns if c in df.columns]]

        # Step 3: Append the modified unique rows to df
        df = pd.concat([df, unique_rows], ignore_index=True)

        print(f"columns after appending {df.columns}")
        print(f"unique rows columns {unique_rows.columns}")
        assert len(df) == 978, f"len(df) {len(df)} == 978"
        
        new_rows = []
        for i, row in df.iterrows():
            for j in range(num_generated_cols):
                colname = f"{cfg.model_generated_potentially_faster_code_col}_{j}" if num_generated_cols > 0 else cfg.model_generated_potentially_faster_code_col
                if row[colname] is None or pd.isna(row[colname]) or pd.isnull(row[colname]):
                    row[f"{colname}_agg_runtime_adjusted"] = float("inf")
                if row[f"{colname}_accuracy"] < cfg.threshold_accuracy:
                    row[f"{colname}_agg_runtime_adjusted"] = float("inf")
                else: 
                    row[f"{colname}_agg_runtime_adjusted"] = row[f"{colname}_agg_runtime"]
            row["fastest_generated_agg_runtime"] = min([row[f"{cfg.model_generated_potentially_faster_code_col}_{j}_agg_runtime_adjusted"] for j in range(num_generated_cols)])
            new_rows.append(row)
            
        df = pd.DataFrame(new_rows)
        
        problem_id_to_fastest_agg_runtime = {}
        problem_id_to_fastest_correctness = {}
        for i, group in df.groupby("problem_id"):
            problem_id_to_fastest_agg_runtime[i] = group["fastest_generated_agg_runtime"].min()
            problem_id_to_fastest_correctness[i] = problem_id_to_fastest_agg_runtime[i] < float("inf")
            
        df["fastest_generated_runtime_over_all_submissions"] = df["problem_id"].apply(lambda x: problem_id_to_fastest_agg_runtime[x])
        df["fastest_generated_speedup_over_all_submissions"] = df[cfg.slow_code_col+"_agg_runtime"] / df["fastest_generated_runtime_over_all_submissions"]
        df["fastest_generated_speedup_over_all_submissions"] = df["fastest_generated_speedup_over_all_submissions"].apply(lambda x: max(1.0, x))
        df["fastest_generated_correctness_over_all_submissions"] = df["problem_id"].apply(lambda x: problem_id_to_fastest_correctness[x])
        
        
        for i in range(1, num_generated_cols+1):
            if num_generated_cols == 0:
                df[f"agg_runtime_best@{i}"] = df[f"{cfg.model_generated_potentially_faster_code_col}_agg_runtime_adjusted"]
                df[f"accuracy_best@{i}"] = df[f"{cfg.model_generated_potentially_faster_code_col}_accuracy"]
                df[f"is_correct_best@{i}"] = df[f"accuracy_best@{i}"] == cfg.threshold_accuracy
            else:
                df[f"agg_runtime_best@{i}"] = df[[f"{cfg.model_generated_potentially_faster_code_col}_{j}_agg_runtime_adjusted" for j in range(i)]].min(axis=1)
                df[f"accuracy_best@{i}"] = df[[f"{cfg.model_generated_potentially_faster_code_col}_{j}_accuracy" for j in range(i)]].max(axis=1)
                df[f"is_correct_best@{i}"] = df[f"accuracy_best@{i}"] == cfg.threshold_accuracy
            df[f"speedup_best@{i}"] = df[cfg.slow_code_col+"_agg_runtime"] / df[f"agg_runtime_best@{i}"]
            df[f"speedup_best@{i}"] = df[f"speedup_best@{i}"].apply(lambda x: max(1.0, x))
            df["speedup_of_fastest_generated_of_all_submissions"] = df[cfg.slow_code_col+"_agg_runtime"] / df["fastest_generated_runtime_over_all_submissions"]
            df["speedup_of_fastest_generated_of_all_submissions"] = df["speedup_of_fastest_generated_of_all_submissions"].apply(lambda x: max(1.0, x))
        
        ## aggregate over all rows
        agg_df = pd.DataFrame(index=[0])
        # agg_df["fastest_generated_runtime_over_all_submissions"] = df["fastest_generated_runtime_over_all_submissions"].mean()
        agg_df["fastest_generated_correctness_over_all_submissions"] = df["fastest_generated_correctness_over_all_submissions"].mean()
        agg_df["fastest_generated_speedup_over_all_submissions"] = df["fastest_generated_speedup_over_all_submissions"].mean()
        # import pdb
        for i in range(1, num_generated_cols+1):
            # pdb.set_trace()
            agg_df[f"mean_accuracy_best@{i}"] = df[f"accuracy_best@{i}"].mean()
            agg_df[f"is_correct_best@{i}"] = df[f"is_correct_best@{i}"].mean()
            agg_df[f"mean_speedup_best@{i}"] = df[f"speedup_best@{i}"].mean()
            for speedup_threshold in [1.10, 1.25, 1.50, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]:
                agg_df[f"percent_programs_speedup_best@{i}>=speedup_threshold_{speedup_threshold}"] = (df[f"speedup_best@{i}"] >= speedup_threshold).mean()
                
        ## add the speedup of tgt_code over src_code and the threshold speedups of tgt_code over src_code
        df["speedup_tgt_over_src"] = df[cfg.slow_code_col+"_agg_runtime"] / df[cfg.reference_code_col+"_agg_runtime"]
        agg_df["mean_speedup_tgt_over_src"] = df["speedup_tgt_over_src"].mean()
        for speedup_threshold in [1.10, 1.25, 1.50, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]:
            agg_df[f"percent_programs_speedup_tgt_over_src>=speedup_threshold_{speedup_threshold}"] = (df["speedup_tgt_over_src"] >= speedup_threshold).mean()
            agg_df[f"percent_programs_speedup_fastest_generated_over_src>=speedup_threshold_{speedup_threshold}"] = (df["speedup_of_fastest_generated_of_all_submissions"] >= speedup_threshold).mean()
        
        ## pretty print out a report 
        
        ## first print out the columns with asterisks separating fields *********
        print("********* Aggregated Results *********")
        for i in range(1, num_generated_cols+1):
            print(f"********* Results Best at {i} Generations *********")
            mean_accuracy = agg_df[f"mean_accuracy_best@{i}"][0]
            mean_speedup = agg_df[f"mean_speedup_best@{i}"][0]
            
            print(f"mean_accuracy_best@{i}: {mean_accuracy}")
            print(f"mean correctness best@{i}: {agg_df[f'is_correct_best@{i}'][0]}")
            print(f"mean_speedup_best@{i}: {mean_speedup} vs. mean_speedup_tgt_over_src: {agg_df['mean_speedup_tgt_over_src'][0]}")
            for speedup_threshold in [1.10, 1.25, 1.50, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]:
                percent_programs = agg_df[f"percent_programs_speedup_best@{i}>=speedup_threshold_{speedup_threshold}"][0]
                percent_programs_tgt_over_src = agg_df[f"percent_programs_speedup_tgt_over_src>=speedup_threshold_{speedup_threshold}"][0]
                print(f"percent_programs_speedup_best@{i}>=speedup_threshold_{speedup_threshold}: {percent_programs} vs. percent_programs_speedup_tgt_over_src>=speedup_threshold_{speedup_threshold}: {percent_programs_tgt_over_src}")
            print("*****************************************")
        print("********* Results Fastest Generated Over All Submissions *********")
        print("mean correctness fastest_generated_over_all_submissions: ", agg_df["fastest_generated_correctness_over_all_submissions"][0])
        print("average fastest_generated_speedup_over_all_submissions: ", agg_df["fastest_generated_speedup_over_all_submissions"][0])
        for speedup_threshold in [1.10, 1.25, 1.50, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]:
            percent_programs = agg_df[f"percent_programs_speedup_fastest_generated_over_src>=speedup_threshold_{speedup_threshold}"][0]
            print(f"percent_programs_speedup_fastest_generated_over_src>=speedup_threshold_{speedup_threshold}: {percent_programs}")
        print("********* End Aggregated Results *********")
            
        return agg_df, df

# global env #: PieEnvironment
global env
env = None

def sigint_handler(signum, frame):
    global env
    print("Ctrl-C pressed, running teardown...")
    if threading.current_thread().name == "MainThread":
        env.teardown()
    print("Teardown complete, exiting...")
    exit(0)

# Set the signal handler for Ctrl+C (SIGINT)
signal.signal(signal.SIGINT, sigint_handler)



def read_inputs_and_prepare_v2(cfg) -> pd.DataFrame:
    """Reads the model generated output, the reference, joins them, and returns a dataframe with the merged data."""
    logging.info(f"Reading reference file from {cfg.reference_file_path}")
    logging.info(f"Reading model generated outputs from {cfg.model_generated_outputs_path}")

    
    gen_df = pd.read_json(
        cfg.model_generated_outputs_path, lines=True, orient="records"
    )
    gen_df = fix_df_columns(gen_df)
    
    logging.info(f"Read {len(gen_df)} rows from {cfg.model_generated_outputs_path}")
    if cfg.is_prompt_based:
        gen_df["slower_program"] = gen_df.apply(
            lambda x: get_input_from_prompt(x), axis=1
        )
    else:
        gen_df["slower_program"] = gen_df[cfg.slow_code_col].apply(lambda x: x.strip())
        
        
    assert (
        cfg.reference_code_col in gen_df.columns
    ), f"Column {cfg.reference_code_col} not found in {cfg.model_generated_outputs_path}"
    merged = gen_df
    
        
    merged = merged[merged[cfg.slow_code_col] != merged[cfg.reference_code_col]]

    assert (
        len(merged) > 0
    ), f"{cfg.slow_code_col} and {cfg.reference_code_col} are the same for all programs"
    
    if cfg.num_problems_to_evaluate != -1:
        merged = merged[: cfg.num_problems_to_evaluate]
    
    
    # if the generated code is a list, then we have multiple generations per input. 
    # we add one column per generation
    if isinstance(merged[cfg.model_generated_potentially_faster_code_col].iloc[0], list) or isinstance(merged[cfg.model_generated_potentially_faster_code_col].iloc[0], pd.Series) or (merged[cfg.model_generated_potentially_faster_code_col].iloc[0][0] == '[' and merged[cfg.model_generated_potentially_faster_code_col].iloc[0][-1] == ']'):
        
        if isinstance(merged[cfg.model_generated_potentially_faster_code_col].iloc[0], str):
            import ast
            merged[cfg.model_generated_potentially_faster_code_col] = merged[cfg.model_generated_potentially_faster_code_col].apply(lambda x: ast.literal_eval(x))
        if isinstance(merged[cfg.model_generated_potentially_faster_code_col].iloc[0], pd.Series):
            merged[cfg.model_generated_potentially_faster_code_col] = merged[cfg.model_generated_potentially_faster_code_col].apply(lambda x: x.tolist())
        num_generations = max(merged[cfg.model_generated_potentially_faster_code_col].apply(lambda x: len(x)).tolist())
        
        for i in range(num_generations):
            merged[f"{cfg.model_generated_potentially_faster_code_col}_{i}"] = merged[cfg.model_generated_potentially_faster_code_col].apply(lambda x: x[i] if i < len(x) else None)
            # so merged will have the same number of columns for all rows, but some rows will have None in some columns (because they have fewer generations)
    else: 
        num_generations = 1
            
    cfg.num_generated_cols = num_generations
    
    return merged

                                       
                                       
def main(cfg):
    # Step 0
    merged = read_inputs_and_prepare_v2(cfg)
    reference_df = pd.read_json(cfg.reference_file_path, lines=True, orient="records")
    
    logging.info(f"Number of programs to evaluate: {len(merged)}")
    logging.info(f"Input column: {cfg.slow_code_col}")
    logging.info(f"Reference column: {cfg.reference_code_col}")
    logging.info(f"Model generated column: {cfg.model_generated_potentially_faster_code_col}")

    # Step 1: Read the inputs 

    # problem_id_to_ground_truths = read_ground_truths(cfg, merged)
    
    # Step 2: Write the inputs to a temporary directory
    
    tempdir = tempfile.TemporaryDirectory()

    ## we need to melt the dataframe from [slow, fast, generated_i] -> column of code_type and column of code
    generated_cols = []
    if isinstance(merged[cfg.model_generated_potentially_faster_code_col].iloc[0], list):
        generated_cols = [colname for colname in merged.columns if colname.startswith(cfg.model_generated_potentially_faster_code_col) and colname[-1].isdigit()]
    else: 
        generated_cols = [cfg.model_generated_potentially_faster_code_col]
    
    logging.info(f"Generated columns: {generated_cols}")
    code_cols = [cfg.slow_code_col, cfg.reference_code_col] + generated_cols
    
    ##PATCH 
    ## rename src_agg_runtime -> src_code_agg_runtime and tgt_agg_runtime -> tgt_code_agg_runtime
    if "src_agg_runtime" in merged.columns and "tgt_agg_runtime" in merged.columns:
        merged = merged.rename(columns={"src_agg_runtime": cfg.slow_code_col+"_agg_runtime", "tgt_agg_runtime": cfg.reference_code_col+"_agg_runtime"})
    
    melted = pd.melt(merged, 
                     value_vars=code_cols,
                     var_name="code_type",
                     value_name="code", 
                     id_vars = [c for c in merged.columns if c not in code_cols])
    
    orig_len = len(melted)
    #drop code na/null
    melted = melted.dropna(subset=["code"])
    
    # sort by "n_tests"
    melted = melted.sort_values(by=["n_tests"], ascending=False)
    
    if not os.path.exists(os.path.join(cfg.output_dir, "test_results.jsonl")):
        # drop any rows where the code length is 0
        melted = melted[melted["code"].apply(lambda x: len(x) > 0)]
        logging.info(f"Dropped {orig_len - len(melted)} rows with NA or empty code")
        
        if not cfg.redo_src_tgt:
            ## remove and cache the rows where code_type == "src_code" or "tgt_code"
            src_tgt_rows = melted[(melted["code_type"] == f"{cfg.slow_code_col}") | (melted["code_type"] == f"{cfg.reference_code_col}")]
            melted = melted[(melted["code_type"] != f"{cfg.slow_code_col}") & (melted["code_type"] != f"{cfg.reference_code_col}")]
            # pdb.set_trace()
        else: 
            ## if we're re-running the src_code and tgt_code, then cache the old agg_runtimes
            orig_src_colname = cfg.slow_code_col.replace("_code", "_agg_runtime")
            orig_tgt_colname = cfg.reference_code_col.replace("_code", "_agg_runtime")
            new_src_colname = cfg.slow_code_col.replace("_code", "_original_agg_runtime")
            new_tgt_colname = cfg.reference_code_col.replace("_code", "_original_agg_runtime")
            melted.rename(columns={orig_src_colname: new_src_colname, orig_tgt_colname: new_tgt_colname}, inplace=True)
        
        print(f"Number of programs to evaluate after dropping NA: {len(melted)}")
        try: 
            if not os.path.exists(cfg.output_dir):
                os.makedirs(cfg.output_dir)
            global env
            env = simulator.make(timeout_seconds_gem5=120, verbose=True, use_logical_cpus=True, port=8888, workers=40, exit_early_on_fail=True)
            ## iterate in batches of cpus_available, env.submit_mutliple_single_submissions() will submit the batch at once
            new_rows = []
            pbar = tqdm(total=len(melted), desc=f"Submitting {len(melted)} programs to evaluate", smoothing=0)
            if cfg.cpus_available == -1: 
                cfg.cpus_available = len(melted)
            # legacy - we used to submit in batches
            batch = melted
            # currently sorting the list of tests in reverse order of length, so that the (potentially) longest tests are run first
            # this will may give more "conservative" estimates of the runtime with tqdm
            results = env.submit_multiple_single_submissions(batch["code"].tolist(),
                                                                [sorted(list(t), reverse=True) for t in batch["tests"].tolist()],
                                                                batch["problem_id"].tolist(),
                                                                "gem5")
            
            # zip the rows and results together
            for (i, row), result in zip(batch.iterrows(), results):
                row["compilation"] = result.compilation
                row["accuracy"] = result.mean_acc
                row["agg_runtime"] = result.agg_runtime
                row["tc2time"] = result.tc2time
                row["tc2stats"] = result.tc2stats # this is a lot of data, toggle if we need all the outputs from gem5's stats.txt
                new_rows.append(row)
            # pbar.update(len(batch))
            melted = pd.DataFrame(new_rows)
            melted.to_json(
                f"{cfg.output_dir}/melted_test_results.jsonl", 
                orient="records",
                lines=True
            )
            env.teardown()
        ## if we get an exception, we still want to teardown the environment because it will likely leave a docker container running
        except Exception as e:
            print(e)
            traceback.print_exc()
            if threading.current_thread().name == "MainThread":
                # global env
                env.teardown()
            raise e
        
        if not cfg.redo_src_tgt:
            ## add back the src_code and tgt_code rows
            melted = pd.concat([melted, src_tgt_rows])
        
        unmelted_df = unmelt_results(melted, cfg)
        
        unmelted_df.to_json(
            f"{cfg.output_dir}/test_results.jsonl",
            orient="records",
            lines=True
        )
    else:
        unmelted_df = pd.read_json(
            f"{cfg.output_dir}/test_results.jsonl",
            orient="records",
            lines=True
        )
    
    agg_df, result_df = report_results(unmelted_df, cfg, reference_df)
    
    agg_df.to_csv(
        f"{cfg.output_dir}/aggregated_results.csv",
        index=False
    )
    
    result_df.to_json(
        f"{cfg.output_dir}/addtl_stats.jsonl",
        orient="records",
        lines=True
    )
    
    print(f"Results written to {cfg.output_dir}")
    

@dataclass
class EvaluationConfig:
    model_generated_outputs_path: str
    output_dir: str
    reference_file_path: str
    is_prompt_based: bool = False
    model_generated_potentially_faster_code_col: str = "generated_answers"
    slow_code_col: str = "src_code"
    reference_code_col: str = "tgt_code"
    cpuset_cpus: Optional[str] = None
    do_eval: bool = False
    cpus_available: int = 1
    num_problems_to_evaluate: int = -1
    threshold_accuracy: float = 1.0
    redo_src_tgt: bool = False
    num_generated_cols: int = None

def load_config(yaml_path: str) -> EvaluationConfig:
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return EvaluationConfig(**config_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)