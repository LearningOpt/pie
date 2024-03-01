"""
Code used for sampling programs based on the text-generation-inference API at https://github.com/huggingface/text-generation-inference

"""


from text_generation import Client
import pandas as pd
from utils.prompter import Prompter
from tqdm import tqdm
import fire
import re

import concurrent.futures

def extract_first_program(text):
    # Look for the main function's start, considering possible non-standard code
    main_start = re.search(r"\b(?:int\s+)?main\b", text)

    if not main_start:
        return text  # Return original if main is not found

    open_braces = 0
    closing_brace_position = -1
    main_function_started = False

    # Start looking for opening brace after the detected main function
    i = main_start.end()

    while i < len(text):
        if text[i] == "{":
            open_braces += 1
            if not main_function_started:
                main_function_started = True

        elif text[i] == "}":
            open_braces -= 1
            if open_braces == 0 and main_function_started:
                closing_brace_position = i
                break

        i += 1

    # If we found a closing brace for the first program
    if closing_brace_position != -1:
        return text[: closing_brace_position + 1]
    else:
        return text  # Return original text if a matching closing brace wasn't found
    

def postprocess(text, prompt_name):
    
    if prompt_name == 'code_opt':
        return extract_first_program(text)
    else:
        return text


def main(
    test_file=None,
    output_file=None,
    do_sample=None,
    num_samples=8,
    max_new_tokens=1000,
    temperature=0.7,
    num_threads=20, # number of threads to use for parallel processing
    prompt_name="code_opt",
):
    # print do_sample
    print(f"do_sample: {do_sample}")
    # print type of do_sample
    print(f"type of do_sample: {type(do_sample)}")

    client = Client("http://127.0.0.1:8080", timeout=100)
    
    prompter = Prompter(template_name=prompt_name)
    
    print(f"prompt_name: {prompt_name}")

    test_df = pd.read_json(test_file, lines=True, orient="records")

    # create results dataframe with src_code column
    results_df = pd.DataFrame(columns=["src_code"])
    results_df["src_code"] = test_df["src_code"]
    # create empty column for completions
    results_df["generated_answers"] = results_df.apply(lambda x: [], axis=1)

    def process_request(index, src_code):
        all_completions = []

        prompt = prompter.generate_prompt(src_code=src_code)

        if do_sample:
            completions = client.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                best_of=num_samples,
            )
        else:
            completions = client.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # best_of=num_samples,
            )
        
        # get all completions from output
        best_of_sequences = [
            completions.details.best_of_sequences[i].generated_text
            for i in range(len(completions.details.best_of_sequences))
        ]
        
        all_programs = [postprocess(completions.generated_text, prompt_name=prompt_name)] + [
            postprocess(best_of_sequences[i], prompt_name=prompt_name)
            for i in range(len(best_of_sequences))
        ]
        
        return index, all_programs

    # Use ThreadPoolExecutor to process in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_index = {executor.submit(process_request, i, row["src_code"]): i for i, row in test_df.iterrows()}
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(test_df)):
            index, all_programs = future.result()
            results_df.at[index, "generated_answers"] = all_programs

    # add generated_answers column to test_df
    test_df["generated_answers"] = results_df["generated_answers"]

    # save test_df to output_file
    test_df.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    fire.Fire(main)
