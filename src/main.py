import sys
import os
import json

def main():
    """Main function to run the program.
    Wraps the main logic of the program.
    """

    # Check if the params file is given as argument
    args = sys.argv
    if len(args) != 2:
        print(f'WARNING: Expected 1 argument but received {len(args) - 1}. Please provide the absolute or relative filepath to \'params_file.txt\'.')
        exit()
    params_file_path = str(args[1])
    if not os.path.isfile(params_file_path):
        print(f'WARNING: The input {params_file_path} is not a valid filepath. Please provide the absolute or relative filepath to \'params_file.txt\'.')
        exit()

    # Read params file
    with open(params_file_path) as f:
        data = f.readlines()
    json_string = ''
    for d in data:
        d = d.strip()
        if not d.startswith('#') and len(d) > 0:
            json_string += d + '\n'
    params = json.loads(json_string)

