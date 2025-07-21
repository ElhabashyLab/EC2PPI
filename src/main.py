import sys
import os
import json
import data_processing.read_params as read_params

def main():
    """Main function,wraps the main logic of the program.
    """

    # Check if the params file is given as argument
    args = sys.argv
    if len(args) != 2:
        raise ValueError(f'WARNING: Expected 1 argument but received {len(args) - 1}. Please provide the absolute or relative filepath to \'params_file.txt\'.')
    params_file_path = args[1]
    # Read the parameters from the params file
    params = read_params.read_params(params_file_path)

if __name__ == "__main__":
    main()


