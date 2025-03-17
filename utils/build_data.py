import random
import typer
from typing_extensions import Annotated

random.seed(42)

PREFIX_ASR = "Transcribe following speech input:"
PREFIX_ASR_SER = "Predict emotion of following speech input and transcribe it. The output format should be like this: '<emotion> transcription'. prompt:"
# PREFIX_UNIFIED = """
# # Task
# From now on, you are an intelligent voice assistant. You need to provide useful, consistent to the dialogue context, emotionally approval natural response to the user's input speech.
# Given user speech and history, you need to identify the emotion, transcribe the user speech, predict appropriate response emotion, and predict appropriate response text according to the context.
# Each dialogue turn is formatted as: '{speaker}: <emotion> text'.
# You must use only 'A' or 'B' for {speaker}.
# The emotion should be one of following 5 emotions: <anger>, <happiness>, <neutral>, <sadness>, <surprise>

# # Examples
# Following examples show example dialogues with emotion and text. The caption in angle brackets indicate emotion of the transcription.

# ## Example 1
# 1. A: <neutral> It just tastes so good, you know?
# 2. B: <happiness> That's awesome! It's always great to find something you enjoy. Do you use it on anything specific, like toast or cooking?
# 3. A: <surprise> I can't believe it's not butter!
# 4. B: <happiness> Oh wow, you're really passionate about this! So, what is it about "I Can't Believe It's Not Butter" that's got you so surprised?

# ## Example 2
# 1. A: <anger> I can't believe it's not butter!
# 2. B: <neutral> Whoa, okay, let's take a deep breath and try to calm down. Are you actually upset that it's not butter? What's really going on here?

# ## Example 3
# 1. A: <neutral> I watched a baseball game on the weekend
# 2. B: <neutral> Oh cool! how was it?
# 3. A: <sadness> The game wasn't bad
# 4. B: <sadness> You don't seem too happy, did your team lose?

# ## Example 4
# 1. A: <neutral> I watched a baseball game on the weekend
# 2. B: <neutral> Oh cool! how was it?
# 3. A: <happiness> The game wasn't bad
# 4. B: <happiness> That's great to hear! Did your favorite team win?

# Here's the prompt:

# """
PREFIX_UNIFIED_NEW = """
# Task
From now on, you are an intelligent voice assistant. You need to provide useful, consistent to the dialogue context, emotionally approval natural response to the user's input speech.
Given user speech and history, you need to identify the emotion, transcribe the user speech, predict appropriate response emotion, and predict appropriate response text according to the context.
Each dialogue turn is formatted as: '{speaker}: <emotion> text'.
The emotion should be one of following 5 emotions: <anger>, <happiness>, <neutral>, <sadness>, <surprise>.
The generated response should vary in emotion and text based on the user's emotion, even if the input text is the same.

Here's the prompt:

"""


def asr_extract_samples_librispeech(data_dir, s2u, num_samples=50000):
    import os

    all_samples = []

    # >> Walk through the dataset directory
    for speaker_id in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_id)
        if not os.path.isdir(speaker_path):
            continue

        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue

            # > Read transcript file
            trans_file = os.path.join(
                chapter_path, f"{speaker_id}-{chapter_id}.trans.txt"
            )
            if not os.path.exists(trans_file):
                continue

            # > Create mapping of utterance_id to transcript
            transcripts = {}
            with open(trans_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        file_id, text = parts
                        text = text.lower()
                        transcripts[file_id] = text

            # > Match audio files with transcripts
            for audio_file in os.listdir(chapter_path):
                if audio_file.endswith(".flac"):
                    file_id = audio_file.rsplit(".", 1)[0]  # Remove .flac extension
                    if file_id in transcripts:
                        all_samples.append(
                            {
                                "audio_path": os.path.join(chapter_path, audio_file),
                                "text": transcripts[file_id],
                            }
                        )

    if num_samples < len(all_samples):
        all_samples = random.sample(all_samples, num_samples)

    formatted_samples = []
    for sample in all_samples:
        units = s2u(sample["audio_path"], merged=True)
        formatted_sample = {
            "prefix": f"{PREFIX_ASR} {units} ",
            "plain_text": f"{sample['text'].strip()}",
        }
        formatted_samples.append(formatted_sample)

    return formatted_samples


def asr_ser_extract_samples_esd(data_dir, s2u, residual=False):
    import os
    import pandas as pd
    from collections import defaultdict
    from pathlib import Path

    map_emo = {
        "Angry": "anger",
        "Happy": "happiness",
        "Neutral": "neutral",
        "Sad": "sadness",
        "Surprise": "surprise",
    }

    all_samples = []
    samples_per_emo = defaultdict(list)
    base_dir = Path(data_dir)

    # >> Walk through the dataset directory
    for speaker_id in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_id)
        if not os.path.isdir(speaker_path):
            continue

        metadata_path = os.path.join(speaker_path, f"{speaker_id}.txt")
        df = pd.read_csv(
            metadata_path, sep="\t", header=None, names=["file_id", "text", "emotion"]
        )

        for idx, row in df.iterrows():
            transcript = row["text"]
            emotion = map_emo[row["emotion"].strip()]
            audio_path = (
                base_dir / speaker_id / row["emotion"].strip() / f"{row['file_id']}.wav"
            )
            all_samples.append(
                {"audio_path": audio_path, "text": transcript, "emotion": emotion}
            )
            samples_per_emo[emotion].append(
                {"audio_path": audio_path, "text": transcript, "emotion": emotion}
            )

    formatted_samples = []
    for sample in all_samples:

        if residual:
            units, residual_length, residual_path = s2u(
                sample["audio_path"], merged=True, residual=True
            )
            formatted_sample = {
                "prefix": f"{PREFIX_ASR_SER} {units} ",
                "plain_text": f"<{sample['emotion']}> {sample['text'].strip()}",
                "residual_length": residual_length,
                "residual_path": residual_path,
            }
        else:
            units = s2u(sample["audio_path"], merged=True)
            formatted_sample = {
                "prefix": f"{PREFIX_ASR_SER} {units} ",
                "plain_text": f"<{sample['emotion']}> {sample['text'].strip()}",
            }

        formatted_samples.append(formatted_sample)

    return formatted_samples


def unified_extract_samples_esd(data_path, synthesized_path, s2u, residual=False):
    import os
    import json
    import pandas as pd
    from collections import defaultdict
    from pathlib import Path

    def clean_string(text):
        import string
        import unicodedata
        import re

        # > First, normalize Unicode characters to their closest ASCII representation
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ASCII", "ignore").decode("ASCII")

        # > Remove all punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # > Remove redundant whitespaces between words
        text = re.sub(r"\s+", " ", text).strip()

        return text

    map_emo = {
        "Angry": "anger",
        "Happy": "happiness",
        "Neutral": "neutral",
        "Sad": "sadness",
        "Surprise": "surprise",
    }

    text_dic = {}

    samples_per_emo = defaultdict(list)
    formatted_samples = []

    base_dir = Path(data_path)
    wav_path_dic = defaultdict(list)

    text_id = 0
    # >> Walk through the dataset directory
    for speaker_id in os.listdir(data_path):
        speaker_path = os.path.join(data_path, speaker_id)
        if not os.path.isdir(speaker_path):
            continue

        metadata_path = os.path.join(speaker_path, f"{speaker_id}.txt")
        df = pd.read_csv(
            metadata_path, sep="\t", header=None, names=["file_id", "text", "emotion"]
        )

        for idx, row in df.iterrows():
            transcript = row["text"].strip()
            emotion = map_emo[row["emotion"].strip()]
            audio_path = (
                base_dir / speaker_id / row["emotion"].strip() / f"{row['file_id']}.wav"
            )
            if clean_string(transcript) not in text_dic.keys():
                text_dic[clean_string(transcript)] = text_id
                text_id += 1

            wav_path_dic[f"{emotion}_{text_dic[clean_string(transcript)]}"].append(
                audio_path
            )

    # >> Walk through the synthesized file
    with open(synthesized_path, "r") as f:
        dialog_idx = -1
        for line in f:
            dialog_idx += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                raise RuntimeError(f"Error loading dialog_idx: {dialog_idx - 1}")
            history_list = sample["history_turns"]
            history = "\n".join(history_list) + "\n"
            history = history.replace("A: ", "user: ").replace("B: ", "EmoSDS: ")

            cur_emo1 = (
                sample["current_turn1"].split("A: ")[1].split(">")[0].split("<")[1]
            )
            cur_emo2 = (
                sample["current_turn2"].split("A: ")[1].split(">")[0].split("<")[1]
            )
            res_emo1 = (
                sample["response_turn1"].split("B: ")[1].split(">")[0].split("<")[1]
            )
            res_emo2 = (
                sample["response_turn2"].split("B: ")[1].split(">")[0].split("<")[1]
            )
            key_cur_emo1 = cur_emo1

            cur_text1 = sample["current_turn1"].split("A: ")[1].split(">")[1].strip()
            cur_text2 = sample["current_turn2"].split("A: ")[1].split(">")[1].strip()
            res_text1 = sample["response_turn1"].split("B: ")[1].split(">")[1].strip()
            res_text2 = sample["response_turn2"].split("B: ")[1].split(">")[1].strip()
            try:
                key_cur_text1 = text_dic[clean_string(cur_text1)]
                key_cur_text2 = text_dic[clean_string(cur_text2)]
            except KeyError:
                continue

            key1 = f"{key_cur_emo1}_{key_cur_text1}"

            wav_list = wav_path_dic[key1]
            if len(wav_list) != 0:
                wav_paths1 = wav_list

                for wav_path in wav_paths1:
                    if residual:
                        units, residual_length, residual_path = s2u(
                            wav_path,
                            merged=True,
                            residual=True,
                        )
                        sample = {
                            "prefix": f"{PREFIX_UNIFIED_NEW}"
                            + f"{history} "
                            + f"user: {units} ",
                            "plain_text": f"<{cur_emo1}> {cur_text1} EmoSDS: <{res_emo1}> {res_text1}",
                            "residual_length": residual_length,
                            "residual_path": residual_path,
                            "dialogue_id": f"{dialog_idx}_0",
                        }
                    else:
                        units = s2u(wav_path, merged=True)
                        sample = {
                            "prefix": f"{PREFIX_UNIFIED_NEW}"
                            + f"{history} "
                            + f"Input: {units} ",
                            "plain_text": f"<{cur_emo1}> {cur_text1} EmoSDS: <{res_emo1}> {res_text1}",
                            "dialogue_id": f"{dialog_idx}_0",
                        }

                    samples_per_emo[cur_emo1].append(sample)
                    formatted_samples.append(sample)

            # continue

            key_cur_emo2 = cur_emo2
            key2 = f"{key_cur_emo2}_{key_cur_text2}"
            
            wav_list = wav_path_dic[key2]
            if len(wav_list) != 0:
                wav_paths2 = wav_list

                for wav_path in wav_paths2:
                    if residual:
                        units, residual_length, residual_path = s2u(
                            wav_path,
                            merged=True,
                            residual=True,
                        )
                        sample = {
                            "prefix": f"{PREFIX_UNIFIED_NEW}"
                            + f"{history} "
                            + f"user: {units} ",
                            "plain_text": f"<{cur_emo2}> {cur_text2} EmoSDS: <{res_emo2}> {res_text2}",
                            "residual_length": residual_length,
                            "residual_path": residual_path,
                            "dialogue_id": f"{dialog_idx}_1",
                        }
                    else:
                        units = s2u(wav_path, merged=True)
                        sample = {
                            "prefix": f"{PREFIX_UNIFIED_NEW}"
                            + f"{history} "
                            + f"Input: {units} ",
                            "plain_text": f"<{cur_emo2}> {cur_text2} EmoSDS: <{res_emo2}> {res_text2}",
                            "dialogue_id": f"{dialog_idx}_1",
                        }

                    samples_per_emo[cur_emo2].append(sample)
                    formatted_samples.append(sample)

    def extract_balanced_samples(
        samples_per_emo,
        total_samples,
        thres,
        selected_dialogues,
        remove_selected_dialogue=False,
    ):
        num_samples_per_emotion = total_samples // len(samples_per_emo)
        remaining_samples = total_samples % len(samples_per_emo)

        selected_samples = []
        for emotion in samples_per_emo:
            n_samples = num_samples_per_emotion + (1 if remaining_samples > 0 else 0)
            if remaining_samples > 0:
                remaining_samples -= 1

            random.shuffle(samples_per_emo[emotion])

            if len(samples_per_emo[emotion]) >= n_samples:
                if remove_selected_dialogue:
                    selected = []
                    selected_dialogues_temp = []
                    for sample in samples_per_emo[emotion]:
                        if len(selected) >= n_samples:
                            break
                        if sample["dialogue_id"] not in selected_dialogues:
                            selected.append(sample)
                            selected_dialogues_temp.append(sample["dialogue_id"])
                            samples_per_emo[emotion].remove(sample)
                    selected_dialogues.update(selected_dialogues_temp)
                else:
                    selected = random.sample(samples_per_emo[emotion], n_samples)
                    for sample in selected:
                        selected_dialogues.add(sample["dialogue_id"])
                        samples_per_emo[emotion].remove(sample)
                selected_samples.extend(selected)
            else:
                if remove_selected_dialogue:
                    selected = []
                    selected_dialogues_temp = []
                    for sample in samples_per_emo[emotion]:
                        if len(selected) >= len(samples_per_emo[emotion]) - thres:
                            break
                        if sample["dialogue_id"] not in selected_dialogues:
                            selected.append(sample)
                            selected_dialogues_temp.append(sample["dialogue_id"])
                            samples_per_emo[emotion].remove(sample)
                    selected_dialogues.update(selected_dialogues_temp)
                else:
                    selected = random.sample(
                        samples_per_emo[emotion], len(samples_per_emo[emotion]) - thres
                    )
                    for sample in selected:
                        selected_dialogues.add(sample["dialogue_id"])
                        samples_per_emo[emotion].remove(sample)
                selected_samples.extend(selected)

        return selected_samples

    selected_dialogues = set()
    test_samples = extract_balanced_samples(
        samples_per_emo, 250, 100, selected_dialogues
    )
    valid_samples = extract_balanced_samples(
        samples_per_emo, 250, 0, selected_dialogues
    )
    train_samples = extract_balanced_samples(
        samples_per_emo, 20000, 0, selected_dialogues, remove_selected_dialogue=True
    )

    def print_emo_stat(samples):
        emotion_stat = defaultdict(int)
        for sample in samples:
            emotion = sample["plain_text"].split(">")[0].split("<")[-1]
            emotion_stat[emotion] += 1

        for emotion, count in sorted(emotion_stat.items()):
            print(f"    {emotion}: {count} samples\n")

    print("emotion distribution in train samples:")
    print_emo_stat(train_samples)
    print("emotion distribution in valid samples:")
    print_emo_stat(valid_samples)
    print("emotion distribution in test samples:")
    print_emo_stat(test_samples)

    return train_samples, valid_samples, test_samples


def save_to_jsonl(samples, output_file):
    import json

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + "\n")


def main(
    task: Annotated[
        str,
        typer.Argument(
            help="Specify task for dataset format. Options: ['asr', 'asr+ser', 'unified']"
        ),
    ],
    librispeech_dir: Annotated[
        str, typer.Option(help="Directory path for LibriSpeech dataset")
    ] = None,
    esd_dir: Annotated[str, typer.Option(help="Directory path for ESD dataset")] = None,
    esd_syn_path: Annotated[
        str, typer.Option(help="File path to esd synthesized dataset")
    ] = None,
    residual: Annotated[
        bool, typer.Option(help="Whether to use residual feature")
    ] = False,
):
    from speech2unit.speech2unit import Speech2UnitCustom
    from asr_ser_split_jsonl_dataset import asr_ser_split_data

    ckpt_dir = "utils/speech2unit/"
    s2u = Speech2UnitCustom(ckpt_dir=ckpt_dir)

    # >> build ASR task dataset
    if task == "asr":

        if librispeech_dir is not None:
            librispeech_samples = asr_extract_samples_librispeech(librispeech_dir, s2u)
            output_path_librispeech = (
                "data/asr/asr_task_librispeech.jsonl"
            )
            save_to_jsonl(librispeech_samples, output_path_librispeech)
            # print(
            #     f"\nLibrispeech samples saved to {output_path_librispeech}: {len(librispeech_samples)}"
            # )
        else:
            print(
                "Skipping LibriSpeech dataset. You can provide path through --librispeech-dir option"
            )

    # >> build ASR+SER task dataset
    elif task == "asr+ser":
        if esd_dir is not None:
            esd_samples = asr_ser_extract_samples_esd(esd_dir, s2u, residual=residual)
        
            output_path_esd = (
                "data/asr_ser/asr_ser_task_esd.jsonl"
            )
            save_to_jsonl(esd_samples, output_path_esd)
            train_samples, valid_samples, test_samples = asr_ser_split_data(jsonl_path=output_path_esd)
            output_path_train = "data/asr_ser/asr_ser_task_esd_train.jsonl"
            output_path_test = "data/asr_ser/asr_ser_task_esd_test.jsonl"
            output_path_valid = "data/asr_ser/asr_ser_task_esd_valid.jsonl"

            save_to_jsonl(train_samples, output_path_train)
            save_to_jsonl(test_samples, output_path_test)
            save_to_jsonl(valid_samples, output_path_valid)
            # print(f"\nESD samples saved to {output_path_esd}: {len(esd_samples)}")
        else:
            print("Skipping ESD dataset. You can provide path through --esd-dir option")

    # >> build unified task dataset
    elif task == "unified":
        if esd_dir is not None:
            if esd_syn_path is None:
                raise RuntimeError(
                    "Please provide ESD synthesized data file path through --esd-syn-path"
                )
            train_samples, valid_samples, test_samples = unified_extract_samples_esd(
                esd_dir, esd_syn_path, s2u, residual=True
            )

            output_path_esd_train = "data/unified/unified_task_esd_train.jsonl"
            output_path_esd_test = "data/unified/unified_task_esd_test.jsonl"
            output_path_esd_valid = "data/unified/unified_task_esd_valid.jsonl"

            save_to_jsonl(train_samples, output_path_esd_train)
            save_to_jsonl(test_samples, output_path_esd_test)
            save_to_jsonl(valid_samples, output_path_esd_valid)
        else:
            print("Skipping ESD dataset. You can provide path through --esd-dir option")


if __name__ == "__main__":
    typer.run(main)