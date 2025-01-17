import random
import typer
from typing_extensions import Annotated


def asr_extract_samples_styletalk(csv_data_path, s2u, num_samples=20000):
    import pandas as pd

    df = pd.read_csv(csv_data_path)

    # >> Extract required columns and create style combinations
    audio_style_data = []
    for _, row in df.iterrows():
        audio_style_data.append(
            (
                row["curr_audio_id"],
                row["curr_text"].replace("B:", "", 1).strip(),  # remove 'B:' part
            )
        )

    # >> If requested samples are more than available data, use all data
    if num_samples > len(audio_style_data):
        sampled_data = audio_style_data
    else:
        sampled_data = random.sample(audio_style_data, num_samples)

    # >> Format data for LLM training
    formatted_samples = []
    for cur_audio_id, cur_text in sampled_data:
        units = s2u(f"data/styletalk/audio/{cur_audio_id}")
        sample = {
            "prefix": f"Transcribe following speech input: {units} > ",
            "plain_text": f"{cur_text.strip()}",
        }
        formatted_samples.append(sample)

    return formatted_samples


def asr_extract_samples_librispeech(data_dir, s2u, num_samples=50000):
    import os
    from pathlib import Path

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

    # >> Randomly sample if we have more data than requested
    if num_samples < len(all_samples):
        all_samples = random.sample(all_samples, num_samples)

    # >> Format samples for LLM training
    formatted_samples = []
    for sample in all_samples:
        units = s2u(sample["audio_path"])
        formatted_sample = {
            "prefix": f"Transcribe following speech input: {units} ",
            "plain_text": f"{sample['text'].strip()}",
        }
        formatted_samples.append(formatted_sample)

    return formatted_samples


def unified_extract_samples_dailytalk(data_path, s2u, num_samples=100, test_size=0.1):
    import os
    from collections import defaultdict
    import json

    with open(os.path.join(data_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    formatted_samples = []
    emotion_stats = defaultdict(int)

    for dialog_idx in metadata.keys():
        dialog_data = metadata[dialog_idx]
        dialog_turns = len(dialog_data)

        dialog_history = {}
        # >> Process consecutive turns to create current-response pairs
        for turn_idx in range(dialog_turns - 1):
            curr_turn = dialog_data[str(turn_idx)]
            next_turn = dialog_data[str(turn_idx + 1)]

            # > Update history
            if turn_idx > 0:
                dialog_history[str(turn_idx - 1)] = dialog_data[str(turn_idx - 1)][
                    "text"
                ]
                if len(dialog_history) > 5:
                    dialog_history.pop(
                        next(iter(dialog_history))
                    )  # removes the first element

            curr_emotion = curr_turn["emotion"]
            resp_emotion = next_turn["emotion"]

            # > Convert "no emotion" to "neutral" for consistency
            curr_emotion = "neutral" if curr_emotion == "no emotion" else curr_emotion
            resp_emotion = "neutral" if resp_emotion == "no emotion" else resp_emotion

            emotion_stats[curr_emotion] += 1

            # > Get audio path
            curr_audio_path = os.path.join(
                data_path,
                "data",
                str(dialog_idx),
                f"{turn_idx}_{curr_turn['speaker']}_d{dialog_idx}.wav",
            )

            # > Build history string
            history = ""
            if turn_idx > 0:
                spk_names = ("Answer", "Input")
                spk_idx = 0
                for turn, text in reversed(dialog_history.items()):
                    history = f"{spk_names[spk_idx]}: {text}\n" + history
                    spk_idx = int(not spk_idx)

            if os.path.exists(curr_audio_path):
                units = s2u(curr_audio_path)

                sample = {
                    "prefix": f"""
# Task
From now on, you are an intelligent voice assistant. You need to provide useful, consistent to the dialogue context, emotionally approval natural response to the user's input speech.
Given user speech and history, you need to transcribe the user speech, identify the speaking style, predict appropriate response style, and predict appropriate response text according to the context.
The speaking style should be one of following 7 styles: anger, disgust, fear, happiness, neutral, sadness, surprise

# Examples
Following examples show example responses to the transcribed input speech with speaking style and history. The caption in angle brackets indicate speaking style of the transcription."

## Example 1
Input: It just tastes so good, you know?
Answer: That's awesome! It's always great to find something you enjoy. Do you use it on anything specific, like toast or cooking?
Input: <surprise> I can't believe it's not butter!
Answer: <happiness> Oh wow, you're really passionate about this! So, what is it about "I Can't Believe It's Not Butter" that's got you so surprised?

## Example 2
Input: <anger> I can't believe it's not butter!
Answer: <neutral> Whoa, okay, let's take a deep breath and try to calm down. Are you actually upset that it's not butter? What's really going on here?

## Example 3
Input: I watched a baseball game on the weekend
Answer: Oh cool! how was it?
Input: <sadness> The game wasn't bad
Answer: <sadness> You don't seem too happy, did your team lose?

## Example 4
Input: I watched a baseball game on the weekend
Answer: Oh cool! how was it?
Input: <happiness> The game wasn't bad
Answer: <happiness> That's great to hear! Did your favorite team win?

Here's the prompt:

"""
                    + f"{history} "
                    + f"Input: {units} ",
                    "plain_text": f"<{curr_emotion}> {curr_turn['text'].strip()} Answer: <{resp_emotion}> {next_turn['text'].strip()}",
                }

                formatted_samples.append(sample)

    # > Print emotion statistics
    print("\nEmotion Statistics:")
    total_samples = sum(emotion_stats.values())
    for emotion, count in sorted(emotion_stats.items()):
        percentage = (count / total_samples) * 100
        print(f"{emotion}: {count} samples ({percentage:.2f}%)")

    # > If requested samples are more than available data, use all data
    if num_samples > len(formatted_samples):
        pass
    else:
        formatted_samples = random.sample(formatted_samples, num_samples)

    # > Calculate number of samples for test set
    num_test = int(num_samples * test_size)
    num_train = num_samples - num_test

    # > Shuffle and split the samples
    random.shuffle(formatted_samples)
    train_samples = formatted_samples[:num_train]
    test_samples = formatted_samples[num_train:]

    return train_samples, test_samples


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
            help="Specify task for dataset format. Options: ['asr', 'unified']"
        ),
    ],
    librispeech_dir: Annotated[
        str, typer.Option(help="Directory path for LibriSpeech dataset")
    ] = None,
    dailytalk_dir: Annotated[
        str, typer.Option(help="Directory path for DailyTalk dataset")
    ] = None,
):
    from speech2unit.speech2unit import Speech2UnitCustom

    styletalk_train_csv = "data/styletalk/train.csv"
    styletalk_eval_csv = "data/styletalk/eval_without_weather_465.csv"

    ckpt_dir = "utils/speech2unit/"
    s2u = Speech2UnitCustom(ckpt_dir=ckpt_dir)

    # >> build asr task dataset
    if task == "asr":

        if librispeech_dir is not None:
            librispeech_samples = asr_extract_samples_librispeech(librispeech_dir, s2u)
            output_path_librispeech = "data/asr_task_librispeech.jsonl"
            save_to_jsonl(librispeech_samples, output_path_librispeech)
            print(
                f"\nLibrispeech samples saved to {output_path_librispeech}: {len(librispeech_samples)}"
            )
        else:
            print(
                "Skipping LibriSpeech dataset. You can provide path through --librispeech-dir option"
            )

        styletalk_samples_train = asr_extract_samples_styletalk(
            styletalk_train_csv, s2u
        )
        styletalk_samples_eval = asr_extract_samples_styletalk(styletalk_eval_csv, s2u)

        output_path_styletalk_train = "data/asr_task_styletalk_train.jsonl"
        output_path_styletalk_eval = "data/asr_task_styletalk_eval.jsonl"

        save_to_jsonl(styletalk_samples_train, output_path_styletalk_train)
        save_to_jsonl(styletalk_samples_eval, output_path_styletalk_eval)

        print(
            f"\nStyleTalk train samples saved to {output_path_styletalk_train}: {len(styletalk_samples_train)}"
        )
        print(
            f"\nStyleTalk eval samples saved to {output_path_styletalk_eval}: {len(styletalk_samples_eval)}"
        )

    # >> build unified task dataset
    elif task == "unified":
        if dailytalk_dir is None:
            raise RuntimeError(
                "Please provide dailytalk dir path through --dailytalk-dir"
            )

        train_samples, test_samples = unified_extract_samples_dailytalk(
            dailytalk_dir, s2u, test_size=0.05
        )

        output_path_dailytalk_train = "data/unified_task_dailytalk_train.jsonl"
        output_path_dailytalk_test = "data/unified_task_dailytalk_test.jsonl"

        save_to_jsonl(train_samples, output_path_dailytalk_train)
        save_to_jsonl(test_samples, output_path_dailytalk_test)

        print(
            f"\nTotal train samples saved to {output_path_dailytalk_train}: {len(train_samples)}"
        )
        print(
            f"\nTotal test samples saved to {output_path_dailytalk_test}: {len(test_samples)}"
        )


if __name__ == "__main__":
    typer.run(main)
