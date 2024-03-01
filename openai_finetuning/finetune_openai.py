import pandas as pd 
import os 
import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import pie_chatgpt
import re
import json
from typing import List, Dict
import yaml 
import logging 
import shutil 
import uuid 
import time 
import json
import os
from time import sleep
from io import StringIO
import openai 



def load_data(train_path, test_path, max_train, max_val):
    df_train = pd.read_json(train_path, lines=True, orient='records')
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_train = df_train[:max_train]
    df_test = pd.read_json(test_path, lines=True, orient='records')
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    df_test = df_test[:max_val]
    return df_train, df_test

        
        
def prepare_output(code_str, max_len=-1, tokenizer=None):
    # "\n+" -> "\n"
    if max_len > 0 and tokenizer: 
        code_str = code_str[:max_len]
    elif max_len > 0 and not tokenizer:
        raise ValueError("max_len > 0 but no tokenizer provided")
    return code_str
        
        
def prepare_dataset(df, src_code_col, tgt_code_col, max_len=-1, tokenizer=None, max_examples=-1):
    df = df.copy()
    if max_examples > 0:
        df = df.sample(frac=1).reset_index(drop=True)
        df = df[:max_examples]
    training_examples = []
    for i, row in df.iterrows():
        src_code = row[src_code_col]
        src_code_formatted = pie_chatgpt.ChatGPTWrapper.prepare_input(src_code)
        tgt_code = row[tgt_code_col]
        tgt_code_formatted = prepare_output(tgt_code, max_len=max_len, tokenizer=tokenizer)
        
        d = [
            {"role": "system", "content": "You are a helpful assistant that can optimize code."},
            {"role": "user", "content": src_code_formatted},
            {"role": "assistant", "content": tgt_code_formatted},
        ]
        training_examples.append({"messages": d})
    return training_examples



def save_dataset(training_examples: List[Dict], file_name: str):
    with open(file_name, 'w') as jsonl_file:
        for example in training_examples:
            jsonl_file.write(json.dumps(example) + '\n')


def register_file_openai(file_path, outpath, sleep_interval=30):
    logger.info(f"Registering file {file_path} to OpenAI")
    file_dict = openai.File.create(
        file=open(file_path, "rb"),
        purpose='fine-tune',
    )
    logger.info(f"File registered with id {file_dict['id']}")
    while file_dict['status'] != 'processed':
        file_dict = openai.File.retrieve(file_dict['id'])
        logger.info(f"File status: {file_dict['status']}")
        with open(outpath, 'w') as json_file:
            json.dump(file_dict, json_file)
        if file_dict['status'] != 'processed':
            logger.info(f"Sleeping for {sleep_interval} seconds")
        sleep(sleep_interval)
    return file_dict
    

def main(input_train_path, input_test_path, max_train, max_val, max_len, tokenizer,output_dir, model_suffix="pie_opt", epochs=1):
    logging.info(f"Input train path: {input_train_path}; epochs: {epochs}")
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    df_train, df_test = load_data(input_train_path, input_test_path, max_train, max_val)
    logger.info(f"Loaded {len(df_train)} training examples and {len(df_test)} test examples")
    training_examples = prepare_dataset(df_train, "src_code", "tgt_code", max_len=max_len, tokenizer=tokenizer)
    if os.path.exists(os.path.join(output_dir, "train.jsonl")):
        unique_id = uuid.uuid4()
        logger.warning(f"File {os.path.join(output_dir, 'train.jsonl')} already exists, copying to {os.path.join(output_dir, f'train_{unique_id}.jsonl')}")
        shutil.copy(os.path.join(output_dir, "train.jsonl"), os.path.join(output_dir, f"train_{unique_id}.jsonl"))
    save_dataset(training_examples, os.path.join(output_dir, "train.jsonl"))
    training_examples = prepare_dataset(df_test, "src_code", "tgt_code", max_len=max_len, tokenizer=tokenizer)
    if os.path.exists(os.path.join(output_dir, "test.jsonl")):
        unique_id = uuid.uuid4()
        logger.warning(f"File {os.path.join(output_dir, 'test.jsonl')} already exists, copying to {os.path.join(output_dir, f'test_{unique_id}.jsonl')}")
        shutil.copy(os.path.join(output_dir, "test.jsonl"), os.path.join(output_dir, f"test_{unique_id}.jsonl"))
    save_dataset(training_examples, os.path.join(output_dir, "test.jsonl"))
    train_data = register_file_openai(os.path.join(output_dir, "train.jsonl"), os.path.join(output_dir, "openai_train_file.json"))
    val_data = register_file_openai(os.path.join(output_dir, "test.jsonl"), os.path.join(output_dir, "openai_val_file.json"))
    train_data, val_data = wait_on_data(train_data, val_data)
    assert train_data['status'] == 'processed'
    assert val_data['status'] == 'processed'
    with open(os.path.join(output_dir, "openai_train_file.json"), 'w') as train_json_file, open(os.path.join(output_dir, "openai_val_file.json"), 'w') as val_json_file:
        json.dump(train_data, train_json_file)
        json.dump(val_data, val_json_file)
    
    model = openai.FineTuningJob.create(
        model = "gpt-3.5-turbo", 
        training_file = train_data['id'],
        validation_file = val_data['id'],
        suffix = model_suffix, 
        hyperparameters = {"n_epochs": epochs}
    )
    logging.info(f"Model {model['id']} created")
    logging.info(f"Model dict: {model}")
    monitor_model(model, output_dir)
    return model     
    
def wait_on_data(train_data, val_data, max_timeout = 600, sleep_interval=10):
    start = time.time()
    while train_data['status'] != 'processed' or val_data['status'] != 'processed':
        train_data = openai.File.retrieve(train_data['id'])
        val_data = openai.File.retrieve(val_data['id'])
        logger.info(f"Train data status: {train_data['status']} status_details: {train_data['status_details']}")
        logger.info(f"Val data status: {val_data['status']}, status_details: {val_data['status_details']}")
        if time.time() - start > max_timeout:
            raise TimeoutError("Timeout waiting for data")
        logger.info(f"Sleeping for {sleep_interval} seconds")
        sleep(sleep_interval)
    return train_data, val_data
    

def get_step_metrics(file_id):
    content = openai.File.download(file_id)
    eval_result = StringIO(content.decode())
    df = pd.read_csv(eval_result, sep=",")
    return df


def handle_get_step_metrics(file_id, output_dir):
    content = openai.File.download(file_id)
    eval_result = StringIO(content.decode())
    try: 
        df = pd.read_csv(eval_result, sep=",")
        df.to_csv(os.path.join(output_dir, f"success_{file_id}.csv"), index=False)
        return df
    except Exception as e:
        error_message = f"Error reading file {file_id}: {e}\n"
        file_content_message = f"File content: {content}\n"
        file_content_decoded_message = f"File content decoded: {content.decode()}\n"
        eval_result_content_message = f"Eval result content: {eval_result.getvalue()}\n"

        with open(os.path.join(output_dir, f"error_{file_id}.txt"), 'w') as error_file:
            error_file.write(error_message)
            error_file.write(file_content_message)
            error_file.write(file_content_decoded_message)
            error_file.write(eval_result_content_message)
        
        logger.error(error_message)
        logger.error(file_content_message)
        logger.error(file_content_decoded_message)
        logger.error(eval_result_content_message)

        return None
    
SAMPLE_CPP_PROGRAM_TO_OPTIMIZE = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char** argv) {
    int n = 1000000;
    int* a = (int*) malloc(n * sizeof(int));
    int* b = (int*) malloc(n * sizeof(int));
    int* c = (int*) malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
    printf("%d", c[0]);
    free(a);
    free(b);
    free(c);
    return 0;
}
"""




def monitor_model(model_dict, output_dir, sleep_interval=30): 
    model = openai.FineTuningJob.retrieve(model_dict['id'])
    logger.info(f"Model status: {model['status']}")
    while model['status'] != 'succeeded':
        model = openai.FineTuningJob.retrieve(model_dict['id'])
        logger.info(f"Model status: {model['status']}")
        if model['status'] != 'succeeded':
            logger.info(f"Sleeping for {sleep_interval} seconds")
        if "result_files" in model:
            for file_id in model['result_files']:
                if file_id != None:
                    result = openai.File.download(file_id)
                    with open(os.path.join(output_dir, f"result_{file_id}.csv"), 'wb') as result_file:
                        result_file.write(result)
                        logging.info(f"Result file {file_id} saved to {os.path.join(output_dir, f'result_{file_id}.json')}")
                    try: 
                        df = pd.read_csv(os.path.join(output_dir, f"result_{file_id}.csv"))
                        last_row = df.iloc[-1]
                        logger.info(f"Last row: {last_row}")
                    except Exception as e:
                        logger.error(f"Error reading file {file_id}: {e}")
                        logger.error(f"File content: {result}")
                        logger.error(f"File content decoded: {result.decode()}")

        with open(os.path.join(output_dir, "openai_model.json"), 'w') as json_file:
            json.dump(model, json_file)
        sleep(sleep_interval)
        
    if "result_files" in model:
        for file_id in model['result_files']:
            if file_id is not None:
                result = openai.File.download(file_id)
                with open(os.path.join(output_dir, f"result_{file_id}.csv"), 'wb') as result_file:  # 'wb'
                    result_file.write(result)
                logging.info(f"Result file {file_id} saved to {os.path.join(output_dir, f'result_{file_id}.json')}")
                
    with open(os.path.join(output_dir, "openai_model.json"), 'w') as json_file:
        json.dump(model, json_file)
    
    # parse the clock time 
    # finished_at = model['finished_at']
    # started_at = model['started_at']
    # total_time = finished_at - started_at
    finished_at = model.get('finished_at', None)
    started_at = model.get('started_at', None)
    if finished_at is not None and started_at is not None:
        total_time = finished_at - started_at
        logging.info(f"Model {model['id']} finished in {total_time / 60} minutes")
    if "trained_tokens" in model:
        logging.info(f"Model {model['id']} trained tokens: {model['trained_tokens']}")
        
    logging.info(f"Model {model['id']} fine-tuned model: {model['fine_tuned_model']}")
    
    
    chat_log = [
        {"role": "system", "content": "You are a helpful assistant that can optimize code."},
        {"role": "user", "content": pie_chatgpt.ChatGPTWrapper.prepare_input(SAMPLE_CPP_PROGRAM_TO_OPTIMIZE)},
    ]
    
    try: 
        response = openai.ChatCompletion.create(
            model=model['fine_tuned_model'],
            messages=chat_log,
            max_tokens=1000,
            temperature=0.0,
        )
        logging.info(f"************************")
        logging.info(f"Input program: {SAMPLE_CPP_PROGRAM_TO_OPTIMIZE}")
        logging.info("************************")
        logging.info(f"Output program: {response['choices'][0]['message']['content']}")
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        logging.error(f"Chat log: {chat_log}")
    
    return model
    

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



if __name__ == "__main__":
    import transformers 
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else: 
        raise ValueError("No config path provided")
    config = load_config(config_path)
    
    openai.api_key = config['api_key']
    if 'organization' in config and config['organization']:
        openai.organization = config['organization']
    
    assert len(config['model_suffix']) > 0 and len(config['model_suffix']) < 19, "model_suffix must be between 1 and 18 characters"
    
    logger = logging.getLogger(__name__)
    ## log date and time
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])
    logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['output_dir'], 'chatgpt_fine_tuning.log')),
        logging.StreamHandler()
    ]
    )
    
    logging.info(f"Config: {config}")
        
    main(
        input_train_path=config['input_train_path'],
        input_test_path=config['input_test_path'],
        max_train=config['max_train'],
        max_val=config['max_val'],
        max_len=config['max_len'],
        tokenizer=tokenizer,
        output_dir=config['output_dir'],
        model_suffix=config['model_suffix'], 
        epochs=config['epochs']
    )

