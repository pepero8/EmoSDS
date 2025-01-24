import json
import argparse
import typer
from typing_extensions import Annotated


def merge_jsonl_files(input_file1, input_file2, output_file):
    """
    Merge two JSONL files by appending lines from the second file to the first file.

    Args:
        input_file1 (str): Path to the first JSONL file
        input_file2 (str): Path to the second JSONL file
        output_file (str): Path to the output merged JSONL file
    """
    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            # >> Process first file
            with open(input_file1, "r", encoding="utf-8") as file1:
                for line in file1:
                    try:
                        json.loads(line.strip())
                        outfile.write(line)
                    except json.JSONDecodeError as e:
                        print(
                            f"Warning: Skipping invalid JSON line in {input_file1}: {e}"
                        )

            # >> Process second file
            with open(input_file2, "r", encoding="utf-8") as file2:
                for line in file2:
                    try:
                        json.loads(line.strip())
                        outfile.write(line)
                    except json.JSONDecodeError as e:
                        print(
                            f"Warning: Skipping invalid JSON line in {input_file2}: {e}"
                        )
    except RuntimeError as e:
        print(e)


def main(
    file1: Annotated[
        str,
        typer.Argument(help="path to first file"),
    ],
    file2: Annotated[
        str,
        typer.Argument(help="path to second file"),
    ],
    out: Annotated[
        str,
        typer.Argument(help="output path"),
    ],
):
    """
    Merge two jsonl files
    """
    merge_jsonl_files(file1, file2, out)
    print(f"Successfully merged files into {out}")


if __name__ == "__main__":
    typer.run(main)
