import pandas as pd
import openai
import random
import tiktoken
from tqdm import tqdm
from typing import List
from concurrent.futures import ThreadPoolExecutor

random.seed(42)


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)
                print(f"\nRetrying after {delay:.2f} seconds.")

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


class ChatGPTWrapper:
    """A Wrapper for ChatGPT model interaction."""

    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
        """
        Calculate the number of tokens in a text string.

        Args:
        - string (str): The text string to be tokenized.
        - encoding_name (str, optional): The encoding name for tokenization. Defaults to "cl100k_base".
        Returns:
        - int: Number of tokens in the string.
        """
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    @staticmethod
    @retry_with_exponential_backoff
    def call_openai_api(
        slow_code_str: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> str:
        """
        Calls the OpenAI API to optimize a given code.

        Args:
        - slow_code_str (str): The code string that needs to be optimized.

        - max_tokens (int, optional): The maximum number of tokens to be used for generation. Defaults to 1024.
        
        - temperature (float, optional): The temperature value for generation. Defaults to 0.0.

        Returns:
        - str: Optimized code returned by the OpenAI API.
        """
        # Initialize the chat log with system and user inputs
        start_chat_log = [
            {"role": "system", "content": "You are a helpful assistant that can optimize code."},
            {"role": "user", "content": ChatGPTWrapper.prepare_input(slow_code_str)},
        ]
        # Call the OpenAI API with the given chat log
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=start_chat_log,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Extract the optimized code from the response
        return response["choices"][0]["message"]["content"]

    @staticmethod
    def prepare_input(slow_code_str: str) -> str:
        """
        Prepares the input for the OpenAI API by framing the code to be optimized.

        Args:
        - slow_code_str (str): The code string that needs to be framed for optimization.

        Returns:
        - str: Formatted input for the OpenAI API.
        """
        prompt = f"""// slower version::

{slow_code_str}

// optimized version of the same code:

"""
        return prompt


QUESTION_PREFIX = "# slower version:\n\n"
ANSWER_PREFIX = "# optimized version of the same code:\n\n"



def main(input_file: str, output_file: str):
    # Read the jsonl file using pandas
    df = pd.read_json(input_file, lines=True)

    # Ensure src_code is in the dataframe
    if 'src_code' not in df.columns:
        raise ValueError("'src_code' column not found in the input file.")
    
    # Optimize code using multiple threads
    df['optimized_code'] = optimize_code_parallel(df['src_code'].tolist())
    
    # Save the dataframe to a new jsonl file
    df.to_json(output_file, orient='records', lines=True)


def optimize_code_parallel(code_list: List[str], max_workers: int = 5) -> List[str]:
    """
    Function to optimize code using multiple threads.
    
    Args:
    - code_list (List[str]): List of code strings to optimize.
    - max_workers (int): Number of worker threads.
    
    Returns:
    - List[str]: List of optimized code strings.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        optimized_code_list = list(tqdm(executor.map(ChatGPTWrapper.call_openai_api, code_list), total=len(code_list)))
    return optimized_code_list

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python pie_chatgpt.py <input_file> <output_file>")
        sys.exit(1)
    main(input_file=sys.argv[1], output_file=sys.argv[2])