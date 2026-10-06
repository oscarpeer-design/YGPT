# This defines the local LLM used and its parameters

# External dependencies
# subprocess: allows running programs (like a local LLM) on a computer:
# We verify files before running to prevent dependency injection attacks
import subprocess
# Path: library used to find file paths on computer
from pathlib import Path

# Internal dependencies
from config import *

# Text file from which we read the model_path
MODEL_PATH_FILE = "model_path.txt"

# Convert sizes from a number in bytes to a string in Gigabytes
def format_size_gb(size_bytes: int) -> str:
    one_gigabyte = 1024 ** 3
    size_gb = size_bytes / one_gigabyte
    return f"{size_gb:.2f} Gigabytes"

# Verify that the provided model path points to an actual local AI model
def validate_model_path(model_path: Path) -> Path | Exception:
    # Check if the model path exists
    if not model_path.exists():
        return Exception(f"Model does not exist at this location: {model_path}")
    # Check if the path points to a regular file
    if not model_path.is_file():
        return Exception(f"Model is not a regular file at this location: {model_path}")
    # Check if the file has the correct extension
    if model_path.suffix != ".gguf":
        return Exception(f"Expected a '.gguf' extension. Instead got {model_path.suffix}")
    # Check for an absolute path (required to prevent runtime issues)
    if not model_path.is_absolute():
        return Exception(f"Model path is not an absolute path: {model_path}")

    # Check that the model size is not too large (avoid system crash)
    model_size = model_path.stat().st_size
    if model_size > MAX_MODEL_SIZE:
        # Convert maximum model size to a readable string
        s_max_size = format_size_gb(MAX_MODEL_SIZE)
        # Convert actual model size to a readable string
        s_model_size = format_size_gb(model_size)
        # Warn that the model is oversized
        return Exception(
            f"""Model file exceeds the supported size limit.
            Expected {s_max_size} but got {s_model_size}
            For model at path {model_path}"""
        )

    # The model is valid
    return model_path

# extract path from text-file set up by user (model_path.txt)
def read_path_from_file() -> Path|Exception:
    # read the lines from the file
    try:
        with open(MODEL_PATH_FILE, 'r', encoding = 'utf-8') as file:
            # read the first line
            line = file.readline()
            # Check the line is not empty
            if line is None or line == "":
                return Exception(f"The first line of the config file, {MODEL_PATH_FILE}, is empty. The first line should be the absolute file path of the local LLM you wish to use.")
            # Get the first line which is a file path
            print(f"IT IS EXPECTED THAT THE FIRST LINE IN THIS FILE IS AN ABSOLUTE FILE PATH FOR A LOCAL LLM: {MODEL_PATH_FILE}\n")
            # Remove line endings
            line = line.strip()
            # Return the line as a path object
            path = Path(line)
            return path

    except Exception as e:
        return Exception(f"The attempt to read the file {MODEL_PATH_FILE} failed. Error: {e}")

# Gets the model name from path and runs model name extraction and file path validation
def get_and_validate_model_path() -> Path|None:
    # read the path from the file
    path = read_path_from_file()

    # check we found an actual path in the file (an Exception will be returned if it doesn't exist where required)
    if isinstance(path, Exception):
        # print the exception and return
        print(path)
        return

    # validate the path
    valid_path = validate_model_path(path)

    # check if the path is valid (an Exception will be returned if it is not)
    if isinstance(valid_path, Exception):
        # print the exception and return
        print(valid_path)
        return
    
    # the path is now read and validated
    return valid_path

# define class to run model
class LocalModel():
    def __init__(self, model_path:Path):
        # initialise model path
        self.model_path = model_path
        # get model name (stem of file path)
        self.model_name = model_path.stem
   
    # Extract response from llama-cli (ignore UI elements)
    def extract_model_response(self, output: str, prompt: str) -> str:
        # Find the point where llama-cli displays the user's prompt
        prompt_marker = f"> {prompt}"
        if prompt_marker not in output:
            return output.strip()
        # Remove everything before the user's prompt
        response = output.split(prompt_marker, 1)[1]
        # Remove llama-cli's exit message
        if "Exiting..." in response:
            response = response.split("Exiting...", 1)[0]
        return response.strip()

    # run query through the model (taking in a prompt and pasting the output text)
    def run_query_through_model(self, prompt: str, n_predict = 1024) -> bool:
        try:
            # Call llama-cli to run local model
            result = subprocess.run(
                [
                    "llama-cli", # Llamma-cli: the program that will run the local model
                    "-m", str(self.model_path), # The path to the local LLM
                     "-p", prompt, # The prompt that will be run
                    "--temp", str(TEMPERATURE), # The temperature
                    "--ctx-size", str(MODEL_CTX), # The context size available
                    "--threads", str(CPU_THREADS_USED), # The number of threads used
                    "--no-display-prompt", # Don't include display prompt (less noise)
                    "--n-predict", str(n_predict), # The number of tokens consumed in the prediction
                    "--single-turn",
                    "--no-show-timings",
                    "--simple-io",
                ],
                text=True,
                check=True,
                capture_output=True,
                timeout=60
            )

            # Print result of query
            response = self.extract_model_response(result.stdout, prompt)
            print(response)
            # We ran a query successfully
            return True

        except Exception as e:
            # Give error message information to user
            print(f"Tried to run the local model at path {self.model_path}. Got an error: {e}")
            # We failed to run a query
            return False
