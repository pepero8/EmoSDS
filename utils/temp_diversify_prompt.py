import json
import typer
from typing_extensions import Annotated

ASR_SER = " Valid emotions are: <anger>, <happiness>, <neutral>, <sadness>, <surprise>."
# ASR_SER = " Valid emotions are: <anger>, <disgust>, <fear>, <happiness>, <neutral>, <sadness>, <surprise>."
# ASR_SER = " Valid emotions are: <unfriendly>, <cheerful>, <sad>, <friendly>, <neutral>."


def diversify_prompt(file, out, prompts_path, asr_ser):

    prompts = []
    with open(prompts_path, "r") as prompts_path:
        for line in prompts_path:
            prompts.append(line)

    with open(file, "r") as f:
        with open(out, "w") as out:
            prompt_idx = 0
            for line in f:
                data = json.loads(line)
                # plain_text = data["plain_text"]
                # common_prefix = data["prefix"].split("Here's the prompt:")[0]
                if asr_ser:
                    # prefix_input = " Valid" + data["prefix"].split(" Valid")[1]
                    prefix_input = data["prefix"].split("prompt:")[1]  # 임시
                else:
                    prefix_input = data["prefix"].split(
                        "Transcribe following speech input: "
                    )[1]
                # prefix_input = data["prefix"].split("Here's the prompt:")[1]
                # new_prefix = NEW_PREFIX + prefix_input
                # new_prefix = prompts[prompt_idx] + prefix_input
                new_prefix = prompts[prompt_idx] + ASR_SER + prefix_input  # 임시
                prompt_idx = (prompt_idx + 1) % len(prompts)
                data["prefix"] = new_prefix

                # cur_emo = plain_text.split(">")[0].split("<")[-1]
                # answer = plain_text.split("Answer: ")[1].strip()
                # new_plain_text = f"<{cur_emo}> Answer: {answer}"

                # data["plain_text"] = new_plain_text
                json.dump(data, out, ensure_ascii=False)
                out.write("\n")


def diversify_prompt_splitted_task(file, out, prompts_path_asr, prompts_path_ser):
    prompts_asr = []
    prompts_ser = []
    with open(prompts_path_asr, "r") as f_asr:
        for line in f_asr:
            prompts_asr.append(line)

    with open(prompts_path_ser, "r") as f_ser:
        for line in f_ser:
            prompts_ser.append(line)

    with open(file, "r") as f:
        with open(out, "w") as out:
            prompt_asr_idx = 0
            prompt_ser_idx = 0
            for line in f:
                data = json.loads(line)
                # plain_text = data["plain_text"]
                # common_prefix = data["prefix"].split("Here's the prompt:")[0]
                task = data["task"]
                if task == "ser":
                    # if asr_ser:
                    try:
                        prefix_input = " Valid" + data["prefix"].split(" Valid")[1]
                    except KeyError:
                        prefix_input = (
                            " Valid" + data["prefix_ser"].split(" Valid")[1]
                        )  # > 임시 (esd 만들 때 ser은 prefix_ser로 해버려서)
                    # prefix_input = data["prefix"].split("prompt:")[1]  # > 임시
                    new_prefix = prompts_ser[prompt_ser_idx] + prefix_input
                    prompt_ser_idx = (prompt_ser_idx + 1) % len(prompts_ser)
                elif task == "asr":
                    prefix_input = data["prefix"].split(
                        "Transcribe following speech input: "
                    )[1]
                    new_prefix = prompts_asr[prompt_asr_idx] + prefix_input
                    prompt_asr_idx = (prompt_asr_idx + 1) % len(prompts_asr)
                # prefix_input = data["prefix"].split("Here's the prompt:")[1]
                # new_prefix = NEW_PREFIX + prefix_input
                # new_prefix = prompts[prompt_idx] + prefix_input
                # new_prefix = prompts[prompt_idx] + ASR_SER + prefix_input  # > 임시
                # prompt_idx = (prompt_idx + 1) % len(prompts)
                data["prefix"] = new_prefix

                # > 임시 (esd 만들 때 ser은 prefix_ser로 해버려서)
                try:
                    del data["prefix_ser"]
                except KeyError:
                    pass

                # cur_emo = plain_text.split(">")[0].split("<")[-1]
                # answer = plain_text.split("Answer: ")[1].strip()
                # new_plain_text = f"<{cur_emo}> Answer: {answer}"

                # data["plain_text"] = new_plain_text
                json.dump(data, out, ensure_ascii=False)
                out.write("\n")


def main(
    file: Annotated[
        str,
        typer.Argument(help="path to dataset"),
    ],
    out: Annotated[
        str,
        typer.Argument(help="output path"),
    ],
    prompts_path: Annotated[
        str,
        typer.Option(help="path to asr prompts txt file"),
    ] = None,
    prompts_path_asr: Annotated[
        str,
        typer.Option(help="path to asr prompts txt file"),
    ] = None,
    prompts_path_ser: Annotated[
        str,
        typer.Option(help="path to ser prompts txt file"),
    ] = None,
    asr_ser: Annotated[
        bool,
        typer.Option(help="whether it's asr+ser dataset"),
    ] = False,
):
    diversify_prompt(file, out, prompts_path, asr_ser)
    # diversify_prompt_splitted_task(file, out, prompts_path_asr, prompts_path_ser)
    print(f"Successfully completed")


if __name__ == "__main__":
    typer.run(main)

# python3 utils/temp_diversify_prompt.py /home/jhwan98/EmoSDS/data/asr/layer7_k2000_merged/asr_task_librispeech.jsonl /home/jhwan98/EmoSDS/data/asr/layer7_k2000_merged/asr_task_librispeech_diverse_prompt.jsonl --prompts-path /home/jhwan98/EmoSDS/data/asr/prompts_for_asr.txt
# python3 utils/temp_diversify_prompt.py /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/asr_ser_task_train_balanced2.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/asr_ser_task_train_balanced2_diverse_prompt.jsonl --prompts-path /home/jhwan98/EmoSDS/data/asr_ser/prompts_for_asr_ser.txt --asr-ser
# python3 utils/temp_diversify_prompt.py /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/asr_ser_task_test_balanced2.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/asr_ser_task_test_balanced2_diverse_prompt.jsonl --prompts-path /home/jhwan98/EmoSDS/data/asr_ser/prompts_for_asr_ser.txt --asr-ser
# python3 utils/temp_diversify_prompt.py /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/asr_ser_task_valid_balanced2.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/asr_ser_task_valid_balanced2_diverse_prompt.jsonl --prompts-path /home/jhwan98/EmoSDS/data/asr_ser/prompts_for_asr_ser.txt --asr-ser

# python3 utils/temp_diversify_prompt.py /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/asr_ser_task_train_balanced_only_esd.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/asr_ser_task_train_balanced_only_esd_diverse_prompt.jsonl --prompts-path /home/jhwan98/EmoSDS/data/asr_ser/prompts_for_asr_ser.txt --asr-ser
# python3 utils/temp_diversify_prompt.py /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/asr_ser_task_test_balanced_only_esd.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/asr_ser_task_test_balanced_only_esd_diverse_prompt.jsonl --prompts-path /home/jhwan98/EmoSDS/data/asr_ser/prompts_for_asr_ser.txt --asr-ser
# python3 utils/temp_diversify_prompt.py /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/asr_ser_task_valid_balanced_only_esd.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/asr_ser_task_valid_balanced_only_esd_diverse_prompt.jsonl --prompts-path /home/jhwan98/EmoSDS/data/asr_ser/prompts_for_asr_ser.txt --asr-ser

# python3 utils/temp_diversify_prompt.py /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/splitted_task/asr_ser_task_train_balanced_splitted_task.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/splitted_task/asr_ser_task_train_balanced_splitted_task_diverse_prompt.jsonl --prompts-path-asr /home/jhwan98/EmoSDS/data/asr/prompts_for_asr.txt --prompts-path-ser /home/jhwan98/EmoSDS/data/asr_ser/prompts_for_ser.txt
# python3 utils/temp_diversify_prompt.py /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/splitted_task/asr_ser_task_test_balanced_splitted_task.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/splitted_task/asr_ser_task_test_balanced_splitted_task_diverse_prompt.jsonl --prompts-path-asr /home/jhwan98/EmoSDS/data/asr/prompts_for_asr.txt --prompts-path-ser /home/jhwan98/EmoSDS/data/asr_ser/prompts_for_ser.txt
# python3 utils/temp_diversify_prompt.py /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/splitted_task/asr_ser_task_valid_balanced_splitted_task.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer7_k2000/merged/splitted_task/asr_ser_task_valid_balanced_splitted_task_diverse_prompt.jsonl --prompts-path-asr /home/jhwan98/EmoSDS/data/asr/prompts_for_asr.txt --prompts-path-ser /home/jhwan98/EmoSDS/data/asr_ser/prompts_for_ser.txt
