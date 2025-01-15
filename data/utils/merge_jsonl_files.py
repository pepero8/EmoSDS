import json
import argparse

def merge_jsonl_files(input_file1, input_file2, output_file):
    """
    Merge two JSONL files by appending lines from the second file to the first file.
    
    Args:
        input_file1 (str): Path to the first JSONL file
        input_file2 (str): Path to the second JSONL file
        output_file (str): Path to the output merged JSONL file
    """
    try:
        # Open output file in write mode
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # Process first file
            with open(input_file1, 'r', encoding='utf-8') as file1:
                for line in file1:
                    # Verify each line is valid JSON
                    try:
                        json.loads(line.strip())
                        outfile.write(line)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line in {input_file1}: {e}")
            
            # Process second file
            with open(input_file2, 'r', encoding='utf-8') as file2:
                for line in file2:
                    # Verify each line is valid JSON
                    try:
                        json.loads(line.strip())
                        outfile.write(line)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line in {input_file2}: {e}")
    except e:
        print(e)
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two JSONL files")
    parser.add_argument("file1", help="Path to first JSONL file")
    parser.add_argument("file2", help="Path to second JSONL file")
    parser.add_argument("output", help="Path to output merged JSONL file")
    
    args = parser.parse_args()
    merge_jsonl_files(args.file1, args.file2, args.output)
    print(f"Successfully merged files into {args.output}")