# confusion matrix 뽑기 위한 input 데이터 추출 코드

import json


def extract_prompt_parts(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            try:
                # Parse the JSON line
                data = json.loads(line.strip())

                # Extract the prefix
                prefix = data.get("prefix", "")

                # Find the part after "Here's the prompt:"
                history_speech = prefix.split("Here's the prompt:")[1]
                # history = history_speech.split("<sosp>")[0]
                # if "Answer" in history or "Input" in history:
                #     last_spk = history.split()[-1]
                speech = "<<Input>>: <sosp>" + history_speech.split("<sosp>")[1]
                # prompt_start = prefix.find("Here's the prompt:")
                # if prompt_start != -1:
                #     # Extract text after "Here's the prompt:"
                #     prompt_text = prefix[
                #         prompt_start + len("Here's the prompt:") :
                #     ].strip()

                #     # Split by "<sosp>"
                #     sosp_parts = prompt_text.split("<sosp>")

                # Write results to output file
                # outfile.write(f"Original Prefix: {prefix}\n")
                # outfile.write(f"Prompt Parts: {sosp_parts}\n\n")
                outfile.write(f"{speech}\n")

            except json.JSONDecodeError:
                print(f"Error parsing line: {line}")


# Example usage
extract_prompt_parts(
    "/home/jhwan98/EmoSDS/data/unified_task_dailytalk_valid_balanced.jsonl",
    "/home/jhwan98/EmoSDS/data/for_confusion_matrix.txt",
)
