import random
import typer
from typing_extensions import Annotated

random.seed(42)

PREFIX_ASR = "Transcribe following speech input:"
PREFIX_ASR_SER = "Predict emotion of following speech input and transcribe it. The output format should be like this: '<emotion> transcription'. prompt:"
# PREFIX_ASR_SER = "Predict emotion of following speech input and transcribe it. Valid emotions are: <anger>, <disgust>, <fear>, <happiness>, <neutral>, <sadness>, <surprise>."
PREFIX_SER = "Predict emotion of following speech input. Valid emotions are: <anger>, <happiness>, <neutral>, <sadness>, <surprise>."
PREFIX_UNIFIED = """
# Task
From now on, you are an intelligent voice assistant. You need to provide useful, consistent to the dialogue context, emotionally approval natural response to the user's input speech.
Given user speech and history, you need to identify the emotion, transcribe the user speech, predict appropriate response emotion, and predict appropriate response text according to the context.
Each dialogue turn is formatted as: '{speaker}: <emotion> text'.
You must use only 'A' or 'B' for {speaker}.
The emotion should be one of following 5 emotions: <anger>, <happiness>, <neutral>, <sadness>, <surprise>

# Examples
Following examples show example dialogues with emotion and text. The caption in angle brackets indicate emotion of the transcription.

## Example 1
1. A: <neutral> It just tastes so good, you know?
2. B: <happiness> That's awesome! It's always great to find something you enjoy. Do you use it on anything specific, like toast or cooking?
3. A: <surprise> I can't believe it's not butter!
4. B: <happiness> Oh wow, you're really passionate about this! So, what is it about "I Can't Believe It's Not Butter" that's got you so surprised?

## Example 2
1. A: <anger> I can't believe it's not butter!
2. B: <neutral> Whoa, okay, let's take a deep breath and try to calm down. Are you actually upset that it's not butter? What's really going on here?

## Example 3
1. A: <neutral> I watched a baseball game on the weekend
2. B: <neutral> Oh cool! how was it?
3. A: <sadness> The game wasn't bad
4. B: <sadness> You don't seem too happy, did your team lose?

## Example 4
1. A: <neutral> I watched a baseball game on the weekend
2. B: <neutral> Oh cool! how was it?
3. A: <happiness> The game wasn't bad
4. B: <happiness> That's great to hear! Did your favorite team win?

Here's the prompt:

"""
PREFIX_UNIFIED_NEW = """
# Task
From now on, you are an intelligent voice assistant. You need to provide useful, consistent to the dialogue context, emotionally approval natural response to the user's input speech.
Given user speech and history, you need to identify the emotion, transcribe the user speech, predict appropriate response emotion, and predict appropriate response text according to the context.
Each dialogue turn is formatted as: '{speaker}: <emotion> text'.
The emotion should be one of following 5 emotions: <anger>, <happiness>, <neutral>, <sadness>, <surprise>.
The generated response should vary in emotion and text based on the user's emotion, even if the input text is the same.

Here's the prompt:

"""


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
        units = s2u(
            f"SpeechGPT/speechgpt/data/styletalk/audio/{cur_audio_id}",
            merged=False,
            downsample=True,
        )
        sample = {
            "prefix": f"{PREFIX_ASR} {units} > ",
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
        units = s2u(sample["audio_path"], merged=True, downsample=False)
        formatted_sample = {
            "prefix": f"{PREFIX_ASR} {units} ",
            "plain_text": f"{sample['text'].strip()}",
        }
        formatted_samples.append(formatted_sample)

    return formatted_samples


def asr_ser_extract_samples_styletalk(
    csv_data_path, s2u, num_samples=20000, residual=False
):
    import pandas as pd
    from collections import defaultdict
    from pathlib import Path

    df_train = pd.read_csv(csv_data_path + "/train.csv")
    df_eval = pd.read_csv(csv_data_path + "/eval.csv")
    df = pd.concat([df_train, df_eval], ignore_index=True)

    # df = pd.read_csv(csv_data_path)

    # >> Extract required columns and create style combinations
    # audio_style_data = []
    base_dir = Path("/shared/NAS_SSD/jhl/futureinternet/styletalk/audio/")

    all_samples = []
    samples_per_emo = defaultdict(list)

    for _, row in df.iterrows():
        # audio_style_data.append(
        #     (
        #         row["curr_audio_id"],
        #         row["curr_emotion"],
        #         row["curr_text"].replace("B:", "", 1).strip(),  # remove 'B:' part
        #     )
        # )

        audio_path = base_dir / row["curr_audio_id"]

        if row["curr_text"].split(": ")[0] == "B":
            transcript = row["curr_text"].replace("B:", "", 1).strip()
        elif row["curr_text"].split(": ")[0] == "A":
            transcript = row["curr_text"].replace("A:", "", 1).strip()

        emotion = row["curr_emotion"]

        all_samples.append(
            {
                "audio_path": audio_path,
                "text": transcript,
                "emotion": emotion,
            }
        )
        samples_per_emo[emotion].append(
            {"audio_path": audio_path, "text": transcript, "emotion": emotion}
        )

    # > Print emotion statistics
    print("\nStyleTalk emotion Statistics:")
    for emotion, sample_list in sorted(samples_per_emo.items()):
        print(f"    {emotion}: {len(sample_list)} samples")

    # >> Format samples for LLM training
    formatted_samples = []
    for sample in all_samples:

        if residual:
            units, residual_length, residual_path = s2u(
                sample["audio_path"], merged=True, downsample=False, residual=True
            )
            formatted_sample = {
                "prefix": f"{PREFIX_ASR_SER} {units} ",
                "plain_text": f"<{sample['emotion']}> {sample['text'].strip()}",
                "residual_length": residual_length,
                "residual_path": residual_path,
            }
        else:
            units = s2u(sample["audio_path"], merged=True, downsample=False)
            formatted_sample = {
                "prefix": f"{PREFIX_ASR_SER} {units} ",
                "plain_text": f"<{sample['emotion']}> {sample['text'].strip()}",
            }
        # formatted_sample_asr = {
        #     "task": "asr",
        #     "prefix": f"{PREFIX_ASR} {units} ",
        #     "plain_text": f"{sample['text'].strip()}",
        # }
        # formatted_sample_ser = {
        #     "task": "ser",
        #     "prefix": f"{PREFIX_SER} {units} ",
        #     "plain_text": f"<{sample['emotion']}>",
        # }

        # formatted_samples.append(formatted_sample_asr)
        # formatted_samples.append(formatted_sample_ser)
        formatted_samples.append(formatted_sample)

    return formatted_samples


def asr_ser_extract_samples_esd(data_dir, s2u, num_samples=50000, residual=False):
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

    # > Print emotion statistics
    print("\nESD emotion Statistics:")
    for emotion, sample_list in sorted(samples_per_emo.items()):
        print(f"    {emotion}: {len(sample_list)} samples")

    # # >> Randomly sample if we have more data than requested
    # if num_samples < len(all_samples):
    #     all_samples = random.sample(all_samples, num_samples)

    # >> Format samples for LLM training
    formatted_samples = []
    for sample in all_samples:

        if residual:
            units, residual_length, residual_path = s2u(
                sample["audio_path"], merged=True, downsample=False, residual=True
            )
            formatted_sample = {
                "prefix": f"{PREFIX_ASR_SER} {units} ",
                "plain_text": f"<{sample['emotion']}> {sample['text'].strip()}",
                "residual_length": residual_length,
                "residual_path": residual_path,
            }
        else:
            units = s2u(sample["audio_path"], merged=True, downsample=False)
            formatted_sample = {
                "prefix": f"{PREFIX_ASR_SER} {units} ",
                "plain_text": f"<{sample['emotion']}> {sample['text'].strip()}",
            }
        # formatted_sample_asr = {
        #     "task": "asr",
        #     "prefix": f"{PREFIX_ASR} {units} ",
        #     "plain_text": f"{sample['text'].strip()}",
        # }
        # formatted_sample_ser = {
        #     "task": "ser",
        #     "prefix": f"{PREFIX_SER} {units} ",
        #     "plain_text": f"<{sample['emotion']}>",
        # }

        # formatted_samples.append(formatted_sample_asr)
        # formatted_samples.append(formatted_sample_ser)
        formatted_samples.append(formatted_sample)

    return formatted_samples


def asr_ser_extract_samples_ravdess(data_dir, s2u, num_samples=50000):
    import os
    import pandas as pd
    from collections import defaultdict
    from pathlib import Path

    map_emo = {
        "01": "neutral",
        "02": "calm",
        "03": "happiness",
        "04": "sadness",
        "05": "anger",
        "06": "fear",
        "07": "disgust",
        "08": "surprise",
    }

    map_text = {
        "01": "Kids are talking by the door",
        "02": "Dogs are sitting by the door",
    }

    all_samples = []
    samples_per_emo = defaultdict(list)
    # base_dir = Path(data_dir)

    # >> Walk through the dataset directory
    for actor_dir in Path(data_dir).glob("*"):
        if not actor_dir.is_dir():
            raise RuntimeError(f"{actor_dir.name} is not a dir")

        for wav_file in actor_dir.glob("*.wav"):
            components = wav_file.stem.split("-")

            if len(components) == 7:
                modality = components[0]
                vocal_channel = components[1]
                emotion = map_emo[components[2]]
                emotional_intensity = components[3]
                transcript = map_text[components[4]]

                if (
                    modality == "03"
                    and emotion != "calm"
                    and vocal_channel == "01"
                    and emotional_intensity == "02"
                ):

                    all_samples.append(
                        {
                            "audio_path": actor_dir / wav_file.name,
                            "text": transcript,
                            "emotion": emotion,
                        }
                    )
                    samples_per_emo[emotion].append(
                        {
                            "audio_path": actor_dir / wav_file.name,
                            "text": transcript,
                            "emotion": emotion,
                        }
                    )
                else:
                    print(
                        f"skipping file -> modality: {modality}, vocal channel: {vocal_channel}, emotion: {emotion}, emotional intensity: {emotional_intensity}, transcript: {transcript}"
                    )

    # > Print emotion statistics
    print("\nravdess emotion Statistics:")
    for emotion, sample_list in sorted(samples_per_emo.items()):
        print(f"    {emotion}: {len(sample_list)} samples")

    # >> Format samples for LLM training
    formatted_samples = []
    for sample in all_samples:
        units = s2u(sample["audio_path"], merged=True, downsample=False)
        # formatted_sample = {
        #     "prefix": f"{PREFIX_ASR_SER} {units} ",
        #     "plain_text": f"<{sample['emotion']}> {sample['text'].strip()}",
        # }

        formatted_sample_asr = {
            "task": "asr",
            "prefix": f"{PREFIX_ASR} {units} ",
            "plain_text": f"{sample['text'].strip()}",
        }
        formatted_sample_ser = {
            "task": "ser",
            "prefix_ser": f"{PREFIX_SER} {units} ",
            "plain_text": f"<{sample['emotion']}>",
        }

        formatted_samples.append(formatted_sample_asr)
        formatted_samples.append(formatted_sample_ser)

    return formatted_samples


def asr_ser_extract_samples_crema(data_dir, s2u, num_samples=50000):
    import os
    import pandas as pd
    from collections import defaultdict
    from pathlib import Path

    map_emo = {
        "NEU": "neutral",
        "HAP": "happiness",
        "SAD": "sadness",
        "ANG": "anger",
        "FEA": "fear",
        "DIS": "disgust",
    }

    map_text = {
        "IEO": "It's eleven o'clock.",
        "TIE": "That is exactly what happened.",
        "IOM": "I'm on my way to the meeting.",
        "IWW": "I wonder what this is about.",
        "TAI": "The airplane is almost full.",
        "MTI": "Maybe tomorrow it will be cold.",
        "IWL": "I would like a new alarm clock",
        "ITH": "I think I have a doctor's appointment.",
        "DFA": "Don't forget a jacket.",
        "ITS": "I think I've seen this before.",
        "TSI": "The surface is slick.",
        "WSI": "We'll stop in a couple of minutes.",
    }

    all_samples = []
    samples_per_emo = defaultdict(list)
    # base_dir = Path(data_dir)

    # >> Walk through the dataset directory
    for wav_file in Path(data_dir).glob("*.wav"):
        components = wav_file.stem.split("_")

        if len(components) == 4:
            # speaker = components[0]
            text = map_text[components[1]]
            emotion = map_emo[components[2]]
            # emotional_intensity = components[3]

            all_samples.append(
                {
                    "audio_path": Path(data_dir) / wav_file.name,
                    "text": text,
                    "emotion": emotion,
                }
            )
            samples_per_emo[emotion].append(
                {
                    "audio_path": Path(data_dir) / wav_file.name,
                    "text": text,
                    "emotion": emotion,
                }
            )
        else:
            raise Exception(f"Wrong filename: {wav_file.name}")

    # > Print emotion statistics
    print("\ncrema emotion Statistics:")
    for emotion, sample_list in sorted(samples_per_emo.items()):
        print(f"    {emotion}: {len(sample_list)} samples")

    # >> Format samples for LLM training
    formatted_samples = []
    for sample in all_samples:
        try:
            units = s2u(sample["audio_path"], merged=True, downsample=False)
        except:
            print(f"Error: {sample['audio_path']}")
            continue
        # formatted_sample = {
        #     "prefix": f"{PREFIX_ASR_SER} {units} ",
        #     "plain_text": f"<{sample['emotion']}> {sample['text'].strip()}",
        # }

        formatted_sample_asr = {
            "task": "asr",
            "prefix": f"{PREFIX_ASR} {units} ",
            "plain_text": f"{sample['text'].strip()}",
        }
        formatted_sample_ser = {
            "task": "ser",
            "prefix_ser": f"{PREFIX_SER} {units} ",
            "plain_text": f"<{sample['emotion']}>",
        }

        formatted_samples.append(formatted_sample_asr)
        formatted_samples.append(formatted_sample_ser)

    return formatted_samples


def asr_ser_extract_samples_dailytalk(data_dir, s2u, num_samples=50000, residual=False):
    import os
    from collections import defaultdict
    import json

    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    samples_per_emo = defaultdict(list)

    formatted_samples = []
    # emotion_stats = defaultdict(int)

    for dialog_idx in metadata.keys():
        dialog_data = metadata[dialog_idx]
        dialog_turns = len(dialog_data)

        # dialog_history = {}
        # >> Process consecutive turns
        for turn_idx in range(dialog_turns - 1):
            curr_turn = dialog_data[str(turn_idx)]
            # next_turn = dialog_data[str(turn_idx + 1)]

            curr_emotion = curr_turn["emotion"]
            # resp_emotion = next_turn["emotion"]

            # > Convert "no emotion" to "neutral" for consistency
            curr_emotion = "neutral" if curr_emotion == "no emotion" else curr_emotion
            # resp_emotion = "neutral" if resp_emotion == "no emotion" else resp_emotion

            # emotion_stats[curr_emotion] += 1

            # > Get audio path
            curr_audio_path = os.path.join(
                data_dir,
                "data",
                str(dialog_idx),
                f"{turn_idx}_{curr_turn['speaker']}_d{dialog_idx}.wav",
            )

            if os.path.exists(curr_audio_path):
                if residual:
                    units, residual_length, residual_path = s2u(
                        curr_audio_path,
                        merged=True,
                        downsample=False,
                        residual=True,
                    )
                    sample = {
                        "prefix": f"{PREFIX_ASR_SER} {units} ",
                        "plain_text": f"<{curr_emotion}> {curr_turn['text'].strip()}",
                        "residual_length": residual_length,
                        "residual_path": residual_path,
                    }
                else:
                    units = s2u(curr_audio_path, merged=True, downsample=False)

                    sample = {
                        "prefix": f"{PREFIX_ASR_SER} {units} ",
                        "plain_text": f"<{curr_emotion}> {curr_turn['text'].strip()}",
                    }

                samples_per_emo[curr_emotion].append(sample)
                formatted_samples.append(sample)

                # sample_asr = {
                #     "task": "asr",
                #     "prefix": f"{PREFIX_ASR} {units} ",
                #     "plain_text": f"{curr_turn['text'].strip()}",
                # }
                # sample_ser = {
                #     "task": "ser",
                #     "prefix": f"{PREFIX_SER} {units} ",
                #     "plain_text": f"<{curr_emotion}>",
                # }

                # formatted_samples.append(sample_asr)

                # if curr_emotion != "fear" and curr_emotion != "disgust":
                #     formatted_samples.append(sample)
                #     samples_per_emo[curr_emotion].append(sample)

    # > Print emotion statistics
    print("\nDailyTalk emotion Statistics:")
    for emotion, sample_list in sorted(samples_per_emo.items()):
        print(f"    {emotion}: {len(sample_list)} samples")

    return formatted_samples


def unified_extract_samples_styletalk(data_path, s2u, residual=False):
    import os
    import pandas as pd
    from collections import defaultdict
    from pathlib import Path

    # with open(os.path.join(data_path, "metadata.json"), "r") as f:
    #     metadata = json.load(f)

    # samples_per_emo = defaultdict(list)

    formatted_samples = []
    emotion_stats = defaultdict(int)

    # df_train = pd.read_csv(csv_data_path + "/train.csv")
    # df_eval = pd.read_csv(csv_data_path + "/eval.csv")
    # df = pd.concat([df_train, df_eval], ignore_index=True)
    df = pd.read_csv(data_path)

    # >> Extract required columns and create style combinations
    # audio_style_data = []
    base_dir = Path("/shared/NAS_SSD/jhl/futureinternet/styletalk/audio/")

    # all_samples = []
    samples_per_emo = defaultdict(list)

    for _, row in df.iterrows():
        # audio_style_data.append(
        #     (
        #         row["curr_audio_id"],
        #         row["curr_emotion"],
        #         row["curr_text"].replace("B:", "", 1).strip(),  # remove 'B:' part
        #     )
        # )
        try:
            if row["curr_text"].split(": ")[0] == "B":
                prev_1 = (
                    row["context"].split("B :")[0].split("A :")[1].strip()
                )  # prev_A1
                prev_2 = (
                    row["context"].split("B :")[1].split("A :")[0].strip()
                )  # prev_B
                prev_3 = (
                    row["context"].split("B :")[1].split("A :")[1].strip()
                )  # prev_A2
            elif row["curr_text"].split(": ")[0] == "A":
                prev_1 = (
                    row["context"].split("A :")[0].split("B :")[1].strip()
                )  # prev_B1
                prev_2 = (
                    row["context"].split("A :")[1].split("B :")[0].strip()
                )  # prev_A
                prev_3 = (
                    row["context"].split("A :")[1].split("B :")[1].strip()
                )  # prev_B2
        except:
            raise Exception(f"context: {row['context']}")

        dialog_id = row["diag_id"]
        audio_path = base_dir / row["curr_audio_id"]

        if row["curr_text"].split(": ")[0] == "B":
            transcript = row["curr_text"].replace("B:", "", 1).strip()
        elif row["curr_text"].split(": ")[0] == "A":
            transcript = row["curr_text"].replace("A:", "", 1).strip()

        res_text = row["res_text"].strip()

        cur_emotion = row["curr_emotion"]
        res_emotion = row["res_emotion"]

        # all_samples.append(
        #     {
        #         "audio_path": audio_path,
        #         "text": transcript,
        #         "emotion": emotion,
        #     }
        # )
        # samples_per_emo[emotion].append(
        #     {"audio_path": audio_path, "text": transcript, "emotion": emotion}
        # )

        if os.path.exists(audio_path):
            if residual:
                units, residual_length, residual_path = s2u(
                    audio_path,
                    merged=True,
                    downsample=False,
                    residual=True,
                )
                sample = {
                    "prefix": f"{PREFIX_UNIFIED_NEW}"
                    + f"EmoSDS: {prev_1} user: {prev_2} EmoSDS: {prev_3} "
                    + f"user: {units} ",
                    "plain_text": f"<{cur_emotion}> {transcript} EmoSDS: <{res_emotion}> {res_text}",
                    "residual_length": residual_length,
                    "residual_path": residual_path,
                    "dialogue_id": dialog_id,
                }
            else:
                print(
                    f"no residual is not supported yet for unified_extract_samples_styletalk"
                )

            samples_per_emo[cur_emotion].append(sample)
            formatted_samples.append(sample)

    # > Print emotion statistics
    print("\nTotal samples: ", len(formatted_samples))
    print("\nstyletalk emotion Statistics:")
    for emotion, sample_list in sorted(samples_per_emo.items()):
        print(f"    {emotion}: {len(sample_list)} samples")

    # def extract_balanced_samples(
    #     samples_per_emo,
    #     total_samples,
    #     thres,
    #     selected_dialogues,
    #     remove_selected_dialogue=False,
    # ):
    #     num_samples_per_emotion = total_samples // len(samples_per_emo)
    #     remaining_samples = total_samples % len(samples_per_emo)

    #     selected_samples = []
    #     for emotion in samples_per_emo:
    #         n_samples = num_samples_per_emotion + (1 if remaining_samples > 0 else 0)
    #         if remaining_samples > 0:
    #             remaining_samples -= 1

    #         if len(samples_per_emo[emotion]) >= n_samples:
    #             if remove_selected_dialogue:
    #                 selected = []
    #                 for sample in samples_per_emo[emotion]:
    #                     if len(selected) >= n_samples:
    #                         break
    #                     if sample["dialogue_idx"] not in selected_dialogues:
    #                         selected.append(sample)
    #                         selected_dialogues.add(sample["dialogue_idx"])
    #                         samples_per_emo[emotion].remove(sample)
    #             else:
    #                 selected = random.sample(samples_per_emo[emotion], n_samples)
    #                 for sample in selected:
    #                     selected_dialogues.add(sample["dialogue_idx"])
    #                     # if remove_selected_dialogue:
    #                     #     if sample["dialogue_idx"] in selected_dialogues:
    #                     #         selected.remove(sample)
    #                     samples_per_emo[emotion].remove(sample)
    #             selected_samples.extend(selected)
    #         else:
    #             # > if we don't have enough, leave #thres samples and take rest
    #             if remove_selected_dialogue:
    #                 selected = []
    #                 for sample in samples_per_emo[emotion]:
    #                     if len(selected) >= len(samples_per_emo[emotion]) - thres:
    #                         break
    #                     if sample["dialogue_idx"] not in selected_dialogues:
    #                         selected.append(sample)
    #                         selected_dialogues.add(sample["dialogue_idx"])
    #                         samples_per_emo[emotion].remove(sample)
    #             else:
    #                 selected = random.sample(
    #                     samples_per_emo[emotion], len(samples_per_emo[emotion]) - thres
    #                 )
    #                 for sample in selected:
    #                     selected_dialogues.add(sample["dialogue_idx"])
    #                     # if remove_selected_dialogue:
    #                     #     if sample["dialogue_idx"] in selected_dialogues:
    #                     #         selected.remove(sample)
    #                     samples_per_emo[emotion].remove(sample)
    #                 selected_samples.extend(selected)
    #             # selected_samples.extend(samples_per_emo[emotion])
    #             # samples_per_emo[emotion] = []

    #     return selected_samples

    # # > If requested samples are more than available data, use all data
    # if num_samples > len(formatted_samples):
    #     pass
    # else:
    #     formatted_samples = random.sample(formatted_samples, num_samples)

    # # > Calculate number of samples for test set
    # num_test = int(num_samples * test_size)
    # num_train = num_samples - num_test

    # # > Shuffle and split the samples
    # random.shuffle(formatted_samples)
    # train_samples = formatted_samples[:num_train]
    # test_samples = formatted_samples[num_train:]

    # selected_dialogues = set()
    # test_samples = extract_balanced_samples(samples_per_emo, 21, 0, selected_dialogues)
    # valid_samples = extract_balanced_samples(
    #     samples_per_emo, 100, 10, selected_dialogues
    # )
    # train_samples = extract_balanced_samples(
    #     samples_per_emo, 2241, 0, selected_dialogues, remove_selected_dialogue=True
    # )

    # def print_emo_stat(samples):
    #     emotion_stat = defaultdict(int)
    #     for sample in samples:
    #         emotion = sample["plain_text"].split(">")[0].split("<")[-1]
    #         emotion_stat[emotion] += 1

    #     for emotion, count in sorted(emotion_stat.items()):
    #         print(f"    {emotion}: {count} samples\n")

    # print("emotion distribution in train samples:")
    # print_emo_stat(train_samples)
    # print("emotion distribution in valid samples:")
    # print_emo_stat(valid_samples)
    # print("emotion distribution in test samples:")
    # print_emo_stat(test_samples)

    # return train_samples, valid_samples, test_samples
    return formatted_samples


# def unified_extract_samples_dailytalk(data_path, s2u, num_samples=100, test_size=0.1):
def unified_extract_samples_dailytalk(data_path, s2u, residual=False):
    import os
    from collections import defaultdict
    import json

    with open(os.path.join(data_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    samples_per_emo = defaultdict(list)

    formatted_samples = []
    emotion_stats = defaultdict(int)

    # # >> Process each dialogues
    for dialog_idx in metadata.keys():
        dialog_data = metadata[dialog_idx]
        dialog_turns = len(dialog_data)

        dialog_history = {}
        #     for turn_idx in range(dialog_turns - 2):
        #         turn = dialog_data[str(turn_idx)]
        #         emotion = turn["emotion"]
        #         emotion = "neutral" if emotion == "no emotion" else curr_emotion
        #         text = turn["text"]
        #         # > Update history
        #         dialog_history[str(turn_idx)] = f"<{emotion}> {text}"

        #     curr_turn = dialog_data[str(dialog_turns - 2)]
        #     resp_turn = dialog_data[str(dialog_turns - 1)]
        #     curr_emotion = curr_turn["emotion"]
        #     resp_emotion = resp_turn["emotion"]

        #     curr_emotion = "neutral" if curr_emotion == "no emotion" else curr_emotion
        #     resp_emotion = "neutral" if resp_emotion == "no emotion" else resp_emotion

        #     emotion_stats[curr_emotion] += 1

        #     curr_audio_path = os.path.join(
        #         data_path,
        #         "data",
        #         str(dialog_idx),
        #         f"{dialog_turns - 2}_{curr_turn['speaker']}_d{dialog_idx}.wav",
        #     )

        #     history = ""
        #     # if turn_idx > 0:
        #     spk_names = ("EmoSDS", "user")
        #     spk_idx = 0
        #     for turn, text in reversed(dialog_history.items()):
        #         history = f"{spk_names[spk_idx]}: {text}\n" + history
        #         spk_idx = int(not spk_idx)

        #     if os.path.exists(curr_audio_path):
        #         if residual:
        #             units, residual_length, residual_path = s2u(
        #                 curr_audio_path, merged=True, downsample=False, residual=True
        #             )

        #             sample = {
        #                 "prefix": f"{PREFIX_UNIFIED_NEW}"
        #                 + f"{history} "
        #                 + f"user: {units} ",
        #                 "plain_text": f"<{curr_emotion}> {curr_turn['text'].strip()} EmoSDS: <{resp_emotion}> {resp_turn['text'].strip()}",
        #                 "residual_length": residual_length,
        #                 "residual_path": residual_path,
        #                 "dialogue_idx": dialog_idx,
        #             }

        #             samples_per_emo[curr_emotion].append(sample)
        #             formatted_samples.append(sample)
        #         else:
        #             raise RuntimeError("no residual currently not supported")

        # >> Process consecutive turns to create current-response pairs
        for turn_idx in range(dialog_turns - 1):
            curr_turn = dialog_data[str(turn_idx)]
            next_turn = dialog_data[str(turn_idx + 1)]

            if (curr_turn["emotion"] == "fear"
                or curr_turn["emotion"] == "disgust"
                or next_turn["emotion"] == "fear"
                or next_turn["emotion"] == "disgust"): # 대화에 fear나 disgust 감정이 들어가는 건 제외
                    break

            # > Update history
            if turn_idx > 0:
                emo = dialog_data[str(turn_idx - 1)]["emotion"]
                emo = "neutral" if emo == "no emotion" else emo
                text = dialog_data[str(turn_idx - 1)]["text"]
                # dialog_history[str(turn_idx - 1)] = dialog_data[str(turn_idx - 1)][
                #     "text"
                # ]
                # dialog_history[str(turn_idx - 1)] = f"<{emo}> {text}"
                dialog_history[str(turn_idx - 1)] = f"{text}"
                # if len(dialog_history) > 5:
                #     dialog_history.pop(
                #         next(iter(dialog_history))
                #     )  # removes the first element

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
                spk_names = ("EmoSDS", "user")
                spk_idx = 0
                for turn, text in reversed(dialog_history.items()):
                    history = f"{spk_names[spk_idx]}: {text}\n" + history
                    spk_idx = int(not spk_idx)

            if os.path.exists(curr_audio_path):
                if residual:
                    units, residual_length, residual_path = s2u(
                        curr_audio_path, merged=True, downsample=False, residual=True
                    )

                    sample = {
                        "prefix": f"{PREFIX_UNIFIED_NEW}"
                        + f"{history} "
                        + f"user: {units} ",
                        "plain_text": f"<{curr_emotion}> {curr_turn['text'].strip()} EmoSDS: <{resp_emotion}> {next_turn['text'].strip()}",
                        "residual_length": residual_length,
                        "residual_path": residual_path,
                        "dialogue_idx": dialog_idx,
                    }

                    samples_per_emo[curr_emotion].append(sample)
                    formatted_samples.append(sample)
                else:
                    raise RuntimeError("no residual currently not supported")

        # units = s2u(curr_audio_path, merged=True, downsample=False)

        # sample = {
        #     "prefix": f"{PREFIX_UNIFIED}" + f"{history} " + f"Input: {units} ",
        #     "plain_text": f"<{curr_emotion}> {curr_turn['text'].strip()} Answer: <{resp_emotion}> {next_turn['text'].strip()}",
        #     "dialogue_idx": dialog_idx,
        # }

        # samples_per_emo[curr_emotion].append(sample)
        # formatted_samples.append(sample)

    # > Print emotion statistics
    print("\nEmotion Statistics:")
    total_samples = sum(emotion_stats.values())
    for emotion, count in sorted(emotion_stats.items()):
        percentage = (count / total_samples) * 100
        print(f"{emotion}: {count} samples ({percentage:.2f}%)")

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
                        if sample["dialogue_idx"] not in selected_dialogues:
                            selected.append(sample)
                            # selected_dialogues.add(sample["dialogue_idx"])
                            selected_dialogues_temp.append(sample["dialogue_idx"])
                            samples_per_emo[emotion].remove(sample)
                    selected_dialogues.update(selected_dialogues_temp)
                else:
                    selected = random.sample(samples_per_emo[emotion], n_samples)
                    for sample in selected:
                        selected_dialogues.add(sample["dialogue_idx"])
                        # if remove_selected_dialogue:
                        #     if sample["dialogue_idx"] in selected_dialogues:
                        #         selected.remove(sample)
                        samples_per_emo[emotion].remove(sample)
                selected_samples.extend(selected)
            else:
                # > if we don't have enough, leave #thres samples and take rest
                if remove_selected_dialogue:
                    selected = []
                    selected_dialogues_temp = []
                    for sample in samples_per_emo[emotion]:
                        if len(selected) >= len(samples_per_emo[emotion]) - thres:
                            break
                        if sample["dialogue_idx"] not in selected_dialogues:
                            selected.append(sample)
                            # selected_dialogues.add(sample["dialogue_idx"])
                            selected_dialogues_temp.append(sample["dialogue_idx"])
                            samples_per_emo[emotion].remove(sample)
                    selected_dialogues.update(selected_dialogues_temp)
                else:
                    selected = random.sample(
                        samples_per_emo[emotion], len(samples_per_emo[emotion]) - thres
                    )
                    for sample in selected:
                        selected_dialogues.add(sample["dialogue_idx"])
                        # if remove_selected_dialogue:
                        #     if sample["dialogue_idx"] in selected_dialogues:
                        #         selected.remove(sample)
                        samples_per_emo[emotion].remove(sample)
                selected_samples.extend(selected)
                # selected_samples.extend(samples_per_emo[emotion])
                # samples_per_emo[emotion] = []

        return selected_samples

    # # > If requested samples are more than available data, use all data
    # if num_samples > len(formatted_samples):
    #     pass
    # else:
    #     formatted_samples = random.sample(formatted_samples, num_samples)

    # # > Calculate number of samples for test set
    # num_test = int(num_samples * test_size)
    # num_train = num_samples - num_test

    # # > Shuffle and split the samples
    # random.shuffle(formatted_samples)
    # test_samples = formatted_samples[:200]
    # valid_samples = formatted_samples[200:300]
    # train_samples = formatted_samples[300:]  # 2241개

    selected_dialogues = set()
    test_samples = extract_balanced_samples(samples_per_emo, 150, 0, selected_dialogues)
    valid_samples = extract_balanced_samples(
        samples_per_emo, 100, 10, selected_dialogues
    )
    train_samples = extract_balanced_samples(
        samples_per_emo, 700, 0, selected_dialogues, remove_selected_dialogue=True
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
    # return train_samples, test_samples


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
            # all_samples.append(
            #     {"audio_path": audio_path, "text": transcript, "emotion": emotion}
            # )
            # samples_per_emo[emotion].append(
            #     {"audio_path": audio_path, "text": transcript, "emotion": emotion}
            # )

    print(f"Total texts found: {text_id}")
    # with open(
    #     "/home/jhwan98/EmoSDS/data/synthesized/unified/esd_all_unique_texts.txt", "w"
    # ) as f:
    #     for key, value in text_dic.items():
    #         f.write(f"{key}\n")
    #     f.close()

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
            # history = history.replace("A: ", "Input: ").replace("B: ", "Answer: ")
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
            # key_cur_emo2 = emo_dic[cur_emo2]

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

            # try:
            wav_list = wav_path_dic[key1]
            if len(wav_list) != 0:
                # continue
                # > randomly sample 3 audios for this emotion-text pair
                # try:
                #     wav_paths1 = random.sample(wav_list, k=3)
                # except ValueError:
                #     wav_paths1 = random.sample(wav_list, k=1)
                wav_paths1 = random.sample(wav_list, k=1)
                # wav_paths1 = wav_list
                # except KeyError:
                # continue
                # except ValueError as e:
                # raise ValueError(f"{e}, key1: {key1}")
                # wav_paths2 = random.sample(wav_path_dic[key2], k=3)

                for wav_path in wav_paths1:
                    # units = s2u(wav_path, merged=True, downsample=False)

                    # sample = {
                    #     "prefix": f"{PREFIX_UNIFIED}"
                    #     + f"{history} "
                    #     + f"Input: {units} ",
                    #     "plain_text": f"<{cur_emo1}> {cur_text1} Answer: <{res_emo1}> {res_text1}",
                    #     "dialogue_id": f"{dialog_idx}_0",
                    # }

                    # samples_per_emo[cur_emo1].append(sample)
                    # formatted_samples.append(sample)

                    if residual:
                        units, residual_length, residual_path = s2u(
                            wav_path,
                            merged=True,
                            downsample=False,
                            residual=True,
                        )
                        sample = {
                            "prefix": f"{PREFIX_UNIFIED_NEW}"
                            + f"{history} "
                            + f"user: {units} ",
                            "plain_text": f"<{cur_emo1}> {cur_text1} EmoSDS: <{res_emo1}> {res_text1}",
                            "residual_length": residual_length,
                            "residual_path": residual_path,
                            # "dialogue_id": f"{dialog_idx}_0",
                            "dialogue_id": f"{dialog_idx}_0",
                        }
                    else:
                        units = s2u(wav_path, merged=True, downsample=False)
                        # formatted_sample = {
                        #     "prefix": f"{PREFIX_ASR_SER} {units} ",
                        #     "plain_text": f"<{sample['emotion']}> {sample['text'].strip()}",
                        # }
                        sample = {
                            "prefix": f"{PREFIX_UNIFIED}"
                            + f"{history} "
                            + f"Input: {units} ",
                            "plain_text": f"<{cur_emo1}> {cur_text1} Answer: <{res_emo1}> {res_text1}",
                            "dialogue_id": f"{dialog_idx}_0",
                        }

                    samples_per_emo[cur_emo1].append(sample)
                    formatted_samples.append(sample)

            continue

            key_cur_emo2 = cur_emo2
            # try:
            # key_cur_emo2 = emo_dic[cur_emo2]
            # except KeyError:
            #     continue
            key2 = f"{key_cur_emo2}_{key_cur_text2}"
            # try:
            wav_list = wav_path_dic[key2]
            if len(wav_list) != 0:
                # continue
                # > randomly sample 3 audios for this emotion-text pair
                # try:
                #     wav_paths2 = random.sample(wav_list, k=3)
                # except:
                #     wav_paths2 = random.sample(wav_list, k=1)
                # wav_paths2 = random.sample(wav_list, k=1)
                wav_paths2 = wav_list
                # except KeyError:
                # continue
                # wav_paths2 = random.sample(wav_path_dic[key2], k=5)

                for wav_path in wav_paths2:
                    # units = s2u(wav_path, merged=True, downsample=False)

                    if residual:
                        units, residual_length, residual_path = s2u(
                            wav_path,
                            merged=True,
                            downsample=False,
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
                        units = s2u(wav_path, merged=True, downsample=False)
                        # formatted_sample = {
                        #     "prefix": f"{PREFIX_ASR_SER} {units} ",
                        #     "plain_text": f"<{sample['emotion']}> {sample['text'].strip()}",
                        # }
                        sample = {
                            "prefix": f"{PREFIX_UNIFIED}"
                            + f"{history} "
                            + f"Input: {units} ",
                            "plain_text": f"<{cur_emo2}> {cur_text2} Answer: <{res_emo2}> {res_text2}",
                            "dialogue_id": f"{dialog_idx}_1",
                        }

                    # sample = {
                    #     "prefix": f"{PREFIX_UNIFIED}"
                    #     + f"{history} "
                    #     + f"Input: {units} ",
                    #     "plain_text": f"<{cur_emo2}> {cur_text2} Answer: <{res_emo2}> {res_text2}",
                    #     "dialogue_id": f"{dialog_idx}_1",
                    # }

                    samples_per_emo[cur_emo2].append(sample)
                    formatted_samples.append(sample)

    # > Print emotion statistics
    print("\nTotal samples: ", len(formatted_samples))
    print("\nesd emotion Statistics:")
    for emotion, sample_list in sorted(samples_per_emo.items()):
        print(f"    {emotion}: {len(sample_list)} samples")
    print(f"\nTotal unique histories: {dialog_idx}\n")

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
                        # if sample["dialogue_id"] not in selected_dialogues:
                        if sample["dialogue_id"] not in selected_dialogues:
                            selected.append(sample)
                            # selected_dialogues.add(sample["dialogue_id"])
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
                # > if we don't have enough, leave #thres samples and take rest
                if remove_selected_dialogue:
                    selected = []
                    selected_dialogues_temp = []
                    for sample in samples_per_emo[emotion]:
                        if len(selected) >= len(samples_per_emo[emotion]) - thres:
                            break
                        if sample["dialogue_id"] not in selected_dialogues:
                            selected.append(sample)
                            # selected_dialogues.add(sample["dialogue_id"])
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
    # test_samples = extract_balanced_samples(
    #     samples_per_emo, 30, 100, selected_dialogues
    # )
    # valid_samples = extract_balanced_samples(
    #     samples_per_emo, 30, 50, selected_dialogues
    # )
    # train_samples = extract_balanced_samples(
    #     samples_per_emo, 1000, 0, selected_dialogues, remove_selected_dialogue=True
    # )
    test_samples = extract_balanced_samples(
        samples_per_emo, 150, 100, selected_dialogues
    )
    # print(len(selected_dialogues))
    valid_samples = extract_balanced_samples(
        samples_per_emo, 100, 0, selected_dialogues
    )
    # print(len(selected_dialogues))
    train_samples = extract_balanced_samples(
        samples_per_emo, 40000, 0, selected_dialogues, remove_selected_dialogue=True
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


def unified_extract_samples_crema(data_path, synthesized_path, s2u):
    import json
    from collections import defaultdict
    from pathlib import Path

    emo_dic = {
        "neutral": "NEU",
        "happiness": "HAP",
        "sadness": "SAD",
        "anger": "ANG",
        "fear": "FEA",
        "disgust": "DIS",
    }
    # emo_dic_rev = dict(zip(emo_dic.values(), emo_dic.keys()))

    text_dic = {
        "It's eleven o'clock.": "IEO",
        "That is exactly what happened.": "TIE",
        "I'm on my way to the meeting.": "IOM",
        "I wonder what this is about.": "IWW",
        "The airplane is almost full.": "TAI",
        "Maybe tomorrow it will be cold.": "MTI",
        "I would like a new alarm clock": "IWL",
        "I think I have a doctor's appointment.": "ITH",
        "Don't forget a jacket.": "DFA",
        "I think I've seen this before.": "ITS",
        "The surface is slick.": "TSI",
        "We'll stop in a couple of minutes.": "WSI",
    }
    # text_dic_rev = dict(zip(text_dic.values(), text_dic.keys()))

    samples_per_emo = defaultdict(list)
    formatted_samples = []

    # all_wavs = []
    wav_path_dic = defaultdict(list)

    # >> Walk through the dataset directory
    for wav_file in Path(data_path).glob("*.wav"):
        components = wav_file.stem.split("_")

        if len(components) == 4:
            text_id = components[1]
            emotion_id = components[2]

            wav_path_dic[f"{emotion_id}_{text_id}"].append(
                Path(data_path) / wav_file.name
            )
        else:
            raise Exception(f"Wrong filename: {wav_file.name}")

    # >> Walk through the synthesized file
    with open(synthesized_path, "r") as f:
        dialog_idx = -1
        for line in f:
            dialog_idx += 1
            sample = json.loads(line)
            history_list = sample["history_turns"]
            history = "\n".join(history_list) + "\n"
            history = history.replace("A: ", "Input: ").replace("B: ", "Answer: ")

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
            key_cur_emo1 = emo_dic[cur_emo1]
            # key_cur_emo2 = emo_dic[cur_emo2]

            cur_text1 = sample["current_turn1"].split("A: ")[1].split(">")[1].strip()
            cur_text2 = sample["current_turn2"].split("A: ")[1].split(">")[1].strip()
            res_text1 = sample["response_turn1"].split("B: ")[1].split(">")[1].strip()
            res_text2 = sample["response_turn2"].split("B: ")[1].split(">")[1].strip()
            key_cur_text1 = text_dic[cur_text1]
            key_cur_text2 = text_dic[cur_text2]

            key1 = f"{key_cur_emo1}_{key_cur_text1}"
            # key2 = f"{key_cur_emo2}_{key_cur_text2}"
            # > randomly sample 5 audios for this emotion-text pair
            wav_paths1 = random.sample(wav_path_dic[key1], k=5)
            # wav_paths2 = random.sample(wav_path_dic[key2], k=3)

            for wav_path in wav_paths1:
                units = s2u(wav_path, merged=True, downsample=False)

                sample = {
                    "prefix": f"{PREFIX_UNIFIED}" + f"{history} " + f"Input: {units} ",
                    "plain_text": f"<{cur_emo1}> {cur_text1} Answer: <{res_emo1}> {res_text1}",
                    "dialogue_id": f"{dialog_idx}_0",
                }

                samples_per_emo[cur_emo1].append(sample)
                formatted_samples.append(sample)

            try:
                key_cur_emo2 = emo_dic[cur_emo2]
            except KeyError:
                continue
            key2 = f"{key_cur_emo2}_{key_cur_text2}"
            wav_paths2 = random.sample(wav_path_dic[key2], k=5)

            for wav_path in wav_paths2:
                units = s2u(wav_path, merged=True, downsample=False)

                sample = {
                    "prefix": f"{PREFIX_UNIFIED}" + f"{history} " + f"Input: {units} ",
                    "plain_text": f"<{cur_emo2}> {cur_text2} Answer: <{res_emo2}> {res_text2}",
                    "dialogue_id": f"{dialog_idx}_1",
                }

                samples_per_emo[cur_emo2].append(sample)
                formatted_samples.append(sample)

    # > Print emotion statistics
    print("\ncrema emotion Statistics:")
    for emotion, sample_list in sorted(samples_per_emo.items()):
        print(f"    {emotion}: {len(sample_list)} samples")

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

            if len(samples_per_emo[emotion]) >= n_samples:
                if remove_selected_dialogue:
                    selected = []
                    for sample in samples_per_emo[emotion]:
                        if len(selected) >= n_samples:
                            break
                        if sample["dialogue_id"] not in selected_dialogues:
                            selected.append(sample)
                            selected_dialogues.add(sample["dialogue_id"])
                            samples_per_emo[emotion].remove(sample)
                else:
                    selected = random.sample(samples_per_emo[emotion], n_samples)
                    for sample in selected:
                        selected_dialogues.add(sample["dialogue_id"])
                        samples_per_emo[emotion].remove(sample)
                selected_samples.extend(selected)
            else:
                # > if we don't have enough, leave #thres samples and take rest
                if remove_selected_dialogue:
                    selected = []
                    for sample in samples_per_emo[emotion]:
                        if len(selected) >= len(samples_per_emo[emotion]) - thres:
                            break
                        if sample["dialogue_id"] not in selected_dialogues:
                            selected.append(sample)
                            selected_dialogues.add(sample["dialogue_id"])
                            samples_per_emo[emotion].remove(sample)
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
    # test_samples = extract_balanced_samples(
    #     samples_per_emo, 50, 100, selected_dialogues
    # )
    # valid_samples = extract_balanced_samples(
    #     samples_per_emo, 50, 50, selected_dialogues
    # )
    train_samples = extract_balanced_samples(
        samples_per_emo, 15000, 0, selected_dialogues, remove_selected_dialogue=False
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
    # print("emotion distribution in valid samples:")
    # print_emo_stat(valid_samples)
    # print("emotion distribution in test samples:")
    # print_emo_stat(test_samples)

    return train_samples  # , valid_samples, test_samples


def unified_extract_samples_ravdess(data_path, synthesized_path, s2u):
    import json
    from collections import defaultdict
    from pathlib import Path

    emo_dic = {
        "neutral": "01",
        "calm": "02",
        "happiness": "03",
        "sadness": "04",
        "anger": "05",
        "fear": "06",
        "disgust": "07",
        "surprise": "08",
    }

    text_dic = {
        "Kids are talking by the door": "01",
        "Dogs are sitting by the door": "02",
    }

    samples_per_emo = defaultdict(list)
    formatted_samples = []

    # all_wavs = []
    wav_path_dic = defaultdict(list)

    # >> Walk through the dataset directory
    for actor_dir in Path(data_path).glob("*"):
        if not actor_dir.is_dir():
            raise RuntimeError(f"{actor_dir.name} is not a dir")

        for wav_file in actor_dir.glob("*.wav"):
            components = wav_file.stem.split("-")

            if len(components) == 7:
                modality = components[0]
                vocal_channel = components[1]
                emotion_id = components[2]
                emotional_intensity = components[3]
                transcript_id = components[4]

                if (
                    modality == "03"
                    and emotion_id != "02"
                    and vocal_channel == "01"
                    # and emotional_intensity == "02"
                ):

                    wav_path_dic[f"{emotion_id}_{transcript_id}"].append(
                        actor_dir / wav_file.name
                    )
                else:
                    print(
                        f"skipping file -> modality: {modality}, vocal channel: {vocal_channel}, emotion: {emotion_id}, emotional intensity: {emotional_intensity}, transcript: {transcript_id}"
                    )

    # >> Walk through the synthesized file
    with open(synthesized_path, "r") as f:
        dialogue_idx = -1
        for line in f:
            dialogue_idx += 1
            sample = json.loads(line)
            history_list = sample["history_turns"]
            history = "\n".join(history_list) + "\n"
            history = history.replace("A: ", "Input: ").replace("B: ", "Answer: ")

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

            key_cur_emo1 = emo_dic[cur_emo1]
            # key_cur_emo2 = emo_dic[cur_emo2]

            cur_text1 = sample["current_turn1"].split("A: ")[1].split(">")[1].strip()
            cur_text2 = sample["current_turn2"].split("A: ")[1].split(">")[1].strip()
            res_text1 = sample["response_turn1"].split("B: ")[1].split(">")[1].strip()
            res_text2 = sample["response_turn2"].split("B: ")[1].split(">")[1].strip()
            key_cur_text1 = text_dic[cur_text1]
            key_cur_text2 = text_dic[cur_text2]

            key1 = f"{key_cur_emo1}_{key_cur_text1}"
            # key2 = f"{key_cur_emo2}_{key_cur_text2}"
            # > randomly sample 3 audios for this emotion-text pair
            wav_paths1 = random.sample(wav_path_dic[key1], k=5)
            # wav_paths2 = random.sample(wav_path_dic[key2], k=3)

            for wav_path in wav_paths1:
                units = s2u(wav_path, merged=True, downsample=False)

                sample = {
                    "prefix": f"{PREFIX_UNIFIED}" + f"{history} " + f"Input: {units} ",
                    "plain_text": f"<{cur_emo1}> {cur_text1} Answer: <{res_emo1}> {res_text1}",
                    "dialogue_id": f"{dialog_idx}_0",
                }

                samples_per_emo[cur_emo1].append(sample)
                formatted_samples.append(sample)

            try:
                key_cur_emo2 = emo_dic[cur_emo2]
            except KeyError:
                continue
            key2 = f"{key_cur_emo2}_{key_cur_text2}"
            wav_paths2 = random.sample(wav_path_dic[key2], k=5)

            for wav_path in wav_paths2:
                units = s2u(wav_path, merged=True, downsample=False)

                sample = {
                    "prefix": f"{PREFIX_UNIFIED}" + f"{history} " + f"Input: {units} ",
                    "plain_text": f"<{cur_emo2}> {cur_text2} Answer: <{res_emo2}> {res_text2}",
                    "dialogue_id": f"{dialog_idx}_1",
                }

                samples_per_emo[cur_emo2].append(sample)
                formatted_samples.append(sample)

    # > Print emotion statistics
    print("\nravdess emotion Statistics:")
    for emotion, sample_list in sorted(samples_per_emo.items()):
        print(f"    {emotion}: {len(sample_list)} samples")

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

            if len(samples_per_emo[emotion]) >= n_samples:
                if remove_selected_dialogue:
                    selected = []
                    for sample in samples_per_emo[emotion]:
                        if len(selected) >= n_samples:
                            break
                        if sample["dialogue_id"] not in selected_dialogues:
                            selected.append(sample)
                            selected_dialogues.add(sample["dialogue_id"])
                            samples_per_emo[emotion].remove(sample)
                else:
                    selected = random.sample(samples_per_emo[emotion], n_samples)
                    for sample in selected:
                        selected_dialogues.add(sample["dialogue_id"])
                        samples_per_emo[emotion].remove(sample)
                selected_samples.extend(selected)
            else:
                # > if we don't have enough, leave #thres samples and take rest
                if remove_selected_dialogue:
                    selected = []
                    for sample in samples_per_emo[emotion]:
                        if len(selected) >= len(samples_per_emo[emotion]) - thres:
                            break
                        if sample["dialogue_id"] not in selected_dialogues:
                            selected.append(sample)
                            selected_dialogues.add(sample["dialogue_id"])
                            samples_per_emo[emotion].remove(sample)
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
    # test_samples = extract_balanced_samples(samples_per_emo, 7, 0, selected_dialogues)
    # valid_samples = extract_balanced_samples(samples_per_emo, 7, 0, selected_dialogues)
    train_samples = extract_balanced_samples(
        samples_per_emo, 15000, 0, selected_dialogues, remove_selected_dialogue=False
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
    # print("emotion distribution in valid samples:")
    # print_emo_stat(valid_samples)
    # print("emotion distribution in test samples:")
    # print_emo_stat(test_samples)

    return train_samples  # , valid_samples, test_samples


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
    styletalk_dir: Annotated[
        str, typer.Option(help="Directory path for StyleTalk dataset")
    ] = None,
    dailytalk_dir: Annotated[
        str, typer.Option(help="Directory path for DailyTalk dataset")
    ] = None,
    esd_dir: Annotated[str, typer.Option(help="Directory path for ESD dataset")] = None,
    esd_syn_path: Annotated[
        str, typer.Option(help="File path to esd synthesized dataset")
    ] = None,
    ravdess_dir: Annotated[
        str, typer.Option(help="Directory path for ravdess dataset")
    ] = None,
    ravdess_syn_path: Annotated[
        str, typer.Option(help="File path to ravdess synthesized dataset")
    ] = None,
    crema_dir: Annotated[
        str, typer.Option(help="Directory path for crema dataset")
    ] = None,
    crema_syn_path: Annotated[
        str, typer.Option(help="File path to crema synthesized dataset")
    ] = None,
    residual: Annotated[
        str, typer.Option(help="File path to crema synthesized dataset")
    ] = False,
):
    from speech2unit.speech2unit import Speech2UnitCustom

    ckpt_dir = "utils/speech2unit/"
    s2u = Speech2UnitCustom(ckpt_dir=ckpt_dir)

    # >> build asr task dataset
    if task == "asr":

        if librispeech_dir is not None:
            librispeech_samples = asr_extract_samples_librispeech(librispeech_dir, s2u)
            output_path_librispeech = (
                "data/asr/layer6_k256_merged/asr_task_librispeech.jsonl"
            )
            save_to_jsonl(librispeech_samples, output_path_librispeech)
            print(
                f"\nLibrispeech samples saved to {output_path_librispeech}: {len(librispeech_samples)}"
            )
        else:
            print(
                "Skipping LibriSpeech dataset. You can provide path through --librispeech-dir option"
            )

        styletalk_train_csv = "SpeechGPT/speechgpt/data/styletalk/train.csv"
        styletalk_eval_csv = (
            "SpeechGPT/speechgpt/data/styletalk/eval_without_weather_465.csv"
        )

        styletalk_samples_train = asr_extract_samples_styletalk(
            styletalk_train_csv, s2u
        )
        styletalk_samples_eval = asr_extract_samples_styletalk(styletalk_eval_csv, s2u)

        # output_path_styletalk_train = (
        #     "data/asr/layer6_k1000/asr_task_styletalk_train.jsonl"
        # )
        # output_path_styletalk_eval = (
        #     "data/asr/layer6_k1000/asr_task_styletalk_eval.jsonl"
        # )

        # save_to_jsonl(styletalk_samples_train, output_path_styletalk_train)
        # save_to_jsonl(styletalk_samples_eval, output_path_styletalk_eval)

        # print(
        #     f"\nStyleTalk train samples saved to {output_path_styletalk_train}: {len(styletalk_samples_train)}"
        # )
        # print(
        #     f"\nStyleTalk eval samples saved to {output_path_styletalk_eval}: {len(styletalk_samples_eval)}"
        # )

    # >> build ASR+SER task dataset
    elif task == "asr+ser":
        if styletalk_dir is not None:
            styletalk_samples = asr_ser_extract_samples_styletalk(
                styletalk_dir, s2u, residual=True
            )
            output_path_styletalk = (
                "data/asr_ser/layer6_k1000/merged/asr_ser_task_styletalk_residual.jsonl"
            )
            save_to_jsonl(styletalk_samples, output_path_styletalk)
            print(
                f"\nStyleTalk samples saved to {output_path_styletalk}: {len(styletalk_samples)}"
            )
        else:
            print(
                "Skipping StyleTalk dataset. You can provide path through --styletalk-dir option"
            )

        if dailytalk_dir is not None:
            dailytalk_samples = asr_ser_extract_samples_dailytalk(
                dailytalk_dir, s2u, residual=True
            )
            output_path_dailytalk = (
                "data/asr_ser/layer6_k1000/merged/asr_ser_task_dailytalk_newresidual.jsonl"
            )
            save_to_jsonl(dailytalk_samples, output_path_dailytalk)
            print(
                f"\nDailytalk samples saved to {output_path_dailytalk}: {len(dailytalk_samples)}"
            )
        else:
            print(
                "Skipping Dailytalk dataset. You can provide path through --dailytalk-dir option"
            )

        if esd_dir is not None:
            esd_samples = asr_ser_extract_samples_esd(esd_dir, s2u, residual=residual)
            output_path_esd = (
                "data/asr_ser/layer6_k1000/merged/asr_ser_task_esd_residual.jsonl"
            )
            save_to_jsonl(esd_samples, output_path_esd)
            print(f"\nESD samples saved to {output_path_esd}: {len(esd_samples)}")
        else:
            print("Skipping ESD dataset. You can provide path through --esd-dir option")

        if ravdess_dir is not None:
            ravdess_samples = asr_ser_extract_samples_ravdess(ravdess_dir, s2u)
            output_path_ravdess = (
                "data/asr_ser/layer6_k2048/merged/asr_ser_task_ravdess.jsonl"
            )
            save_to_jsonl(ravdess_samples, output_path_ravdess)
            print(
                f"\nravdess samples saved to {output_path_ravdess}: {len(ravdess_samples)}"
            )
        else:
            print(
                "Skipping ravdess dataset. You can provide path through --ravdess-dir option"
            )

        if crema_dir is not None:
            crema_samples = asr_ser_extract_samples_crema(crema_dir, s2u)
            output_path_crema = (
                "data/asr_ser/layer6_k2048/merged/asr_ser_task_crema.jsonl"
            )
            save_to_jsonl(crema_samples, output_path_crema)
            print(f"\ncrema samples saved to {output_path_crema}: {len(crema_samples)}")
        else:
            print(
                "Skipping crema dataset. You can provide path through --crema-dir option"
            )

    # >> build unified task dataset
    elif task == "unified":
        if styletalk_dir is not None:
            # train_samples, valid_samples, test_samples = (
            #     unified_extract_samples_styletalk(styletalk_dir, s2u)
            # )
            train_csv = styletalk_dir + "/train.csv"
            eval_csv = styletalk_dir + "/eval.csv"

            train_samples = unified_extract_samples_styletalk(
                train_csv, s2u, residual=True
            )
            valid_samples = unified_extract_samples_styletalk(
                eval_csv, s2u, residual=True
            )

            output_path_styletalk_train = "data/unified/layer6_k1000_merged/unified_task_styletalk_train_balanced.jsonl"
            # output_path_styletalk_test = "data/unified/layer7_k2000_merged2/unified_task_styletalk_test_balanced.jsonl"
            output_path_styletalk_valid = "data/unified/layer6_k1000_merged/unified_task_styletalk_valid_balanced.jsonl"

            save_to_jsonl(train_samples, output_path_styletalk_train)
            # save_to_jsonl(test_samples, output_path_styletalk_test)
            save_to_jsonl(valid_samples, output_path_styletalk_valid)

            print(
                f"\nTotal styletalk train samples saved to {output_path_styletalk_train}: {len(train_samples)}"
            )
            # print(
            #     f"\nTotal styletalk test samples saved to {output_path_styletalk_test}: {len(test_samples)}"
            # )
            print(
                f"\nTotal styletalk valid samples saved to {output_path_styletalk_valid}: {len(valid_samples)}"
            )
        else:
            print(
                "Skipping Styletalk dataset. You can provide path through --styletalk-dir option"
            )

        if dailytalk_dir is not None:
            train_samples, valid_samples, test_samples = (
                unified_extract_samples_dailytalk(dailytalk_dir, s2u, residual=True)
            )
            # train_samples, test_samples = (
            #     unified_extract_samples_dailytalk(dailytalk_dir, s2u, residual=True)
            # )
            # train_samples = unified_extract_samples_dailytalk(dailytalk_dir, s2u)

            output_path_dailytalk_train = "data/unified/layer6_k1000_merged/unified_task_dailytalk_train_balanced_newresidual_2000.jsonl"
            output_path_dailytalk_test = "data/unified/layer6_k1000_merged/unified_task_dailytalk_test_balanced_newresidual_2000.jsonl"
            output_path_dailytalk_valid = "data/unified/layer6_k1000_merged/unified_task_dailytalk_valid_balanced_newresidual_2000.jsonl"

            save_to_jsonl(train_samples, output_path_dailytalk_train)
            save_to_jsonl(test_samples, output_path_dailytalk_test)
            save_to_jsonl(valid_samples, output_path_dailytalk_valid)

            print(
                f"\nTotal dailytalk train samples saved to {output_path_dailytalk_train}: {len(train_samples)}"
            )
            print(
                f"\nTotal dailytalk test samples saved to {output_path_dailytalk_test}: {len(test_samples)}"
            )
            print(
                f"\nTotal dailytalk valid samples saved to {output_path_dailytalk_valid}: {len(valid_samples)}"
            )
        else:
            # raise RuntimeError(
            #     "Please provide dailytalk dir path through --dailytalk-dir"
            # )
            print(
                "Skipping Dailytalk dataset. You can provide path through --dailytalk-dir option"
            )

        if esd_dir is not None:
            if esd_syn_path is None:
                raise RuntimeError(
                    "Please provide ESD synthesized data file path through --esd-syn-path"
                )
            train_samples, valid_samples, test_samples = unified_extract_samples_esd(
                esd_dir, esd_syn_path, s2u, residual=True
            )
            # train_samples, valid_samples = unified_extract_samples_esd(
            #     esd_dir, esd_syn_path, s2u, residual=True
            # )

            output_path_esd_train = "data/unified/layer6_k1000_merged/unified_task_esd_train_balanced_type3_20250216.jsonl"
            output_path_esd_test = "data/unified/layer6_k1000_merged/unified_task_esd_test_balanced_type3_20250216.jsonl"
            output_path_esd_valid = "data/unified/layer6_k1000_merged/unified_task_esd_valid_balanced_type3_20250216.jsonl"

            save_to_jsonl(train_samples, output_path_esd_train)
            save_to_jsonl(test_samples, output_path_esd_test)
            save_to_jsonl(valid_samples, output_path_esd_valid)

            print(
                f"\nTotal esd train samples saved to {output_path_esd_train}: {len(train_samples)}"
            )
            print(
                f"\nTotal esd test samples saved to {output_path_esd_test}: {len(test_samples)}"
            )
            print(
                f"\nTotal esd valid samples saved to {output_path_esd_valid}: {len(valid_samples)}"
            )
        else:
            # raise RuntimeError(
            #     "Please provide dailytalk dir path through --dailytalk-dir"
            # )
            print("Skipping ESD dataset. You can provide path through --esd-dir option")

        if crema_dir is not None:
            if crema_syn_path is None:
                raise RuntimeError(
                    "Please provide CREMA-D synthesized data file path through --crema-syn-path"
                )
            # train_samples, valid_samples, test_samples = unified_extract_samples_crema(
            #     crema_dir, crema_syn_path, s2u
            # )
            train_samples = unified_extract_samples_crema(
                crema_dir, crema_syn_path, s2u
            )
            output_path_crema_train = "data/unified/layer7_k2000_merged2/unified_task_crema_train_balanced.jsonl"
            # output_path_crema_test = "data/unified/layer7_k2000_merged2/unified_task_crema_test_balanced.jsonl"
            # output_path_crema_valid = "data/unified/layer7_k2000_merged2/unified_task_crema_valid_balanced.jsonl"
            save_to_jsonl(train_samples, output_path_crema_train)
            # save_to_jsonl(test_samples, output_path_crema_test)
            # save_to_jsonl(valid_samples, output_path_crema_valid)
            print(
                f"\nTotal crema train samples saved to {output_path_crema_train}: {len(train_samples)}"
            )
            # print(
            #     f"\nTotal crema test samples saved to {output_path_crema_test}: {len(test_samples)}"
            # )
            # print(
            #     f"\nTotal crema valid samples saved to {output_path_crema_valid}: {len(valid_samples)}"
            # )
        else:
            # raise RuntimeError("Please provide CREMA-D dir path through --crema-dir")
            print(
                "Skipping crema dataset. You can provide path through --crema-dir option"
            )

        if ravdess_dir is not None:
            if ravdess_syn_path is None:
                raise RuntimeError(
                    "Please provide ravdess synthesized data file path through --ravdess-syn-path"
                )
            # train_samples, valid_samples, test_samples = (
            #     unified_extract_samples_ravdess(ravdess_dir, ravdess_syn_path, s2u)
            # )
            train_samples = unified_extract_samples_ravdess(
                ravdess_dir, ravdess_syn_path, s2u
            )
            output_path_ravdess_train = "data/unified/layer7_k2000_merged2/unified_task_ravdess_train_balanced.jsonl"
            # output_path_ravdess_test = "data/unified/layer7_k2000_merged2/unified_task_ravdess_test_balanced.jsonl"
            # output_path_ravdess_valid = "data/unified/layer7_k2000_merged2/unified_task_ravdess_valid_balanced.jsonl"
            save_to_jsonl(train_samples, output_path_ravdess_train)
            # save_to_jsonl(test_samples, output_path_ravdess_test)
            # save_to_jsonl(valid_samples, output_path_ravdess_valid)
            print(
                f"\nTotal ravdess train samples saved to {output_path_ravdess_train}: {len(train_samples)}"
            )
            # print(
            #     f"\nTotal ravdess test samples saved to {output_path_ravdess_test}: {len(test_samples)}"
            # )
            # print(
            #     f"\nTotal ravdess valid samples saved to {output_path_ravdess_valid}: {len(valid_samples)}"
            # )
        else:
            # raise RuntimeError("Please provide ravdess dir path through --ravdess-dir")
            print(
                "Skipping ravdess dataset. You can provide path through --ravdess-dir option"
            )


if __name__ == "__main__":
    typer.run(main)
