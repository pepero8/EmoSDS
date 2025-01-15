import pandas as pd
import random
import json
import sys
sys.path.append("/home/jhwan98/EmoSDS/SpeechGPT/speechgpt")
from utils.speech2unit.speech2unit import Speech2UnitCustom


def extract_samples(csv_data_path, s2u, num_samples=100):
    """
    Extract audio IDs, styles, response texts from CSV data and format for LLM training

    Parameters:
    csv_data (str): CSV content path
    num_samples (int): Number of samples to extract

    Returns:
    list: List of dictionaries containing formatted prompts and responses
    """
    df = pd.read_csv(csv_data_path)

    # Extract required columns and create style combinations
    audio_style_data = []
    for _, row in df.iterrows():
        audio_style_data.append(
            (
                row["curr_audio_id"],
                row["curr_text"].replace("B:", "", 1).strip(),  # remove 'B:' part
            )
        )

    # If requested samples are more than available data, use all data
    if num_samples > len(audio_style_data):
        sampled_data = audio_style_data
    else:
        # Randomly sample the data
        sampled_data = random.sample(audio_style_data, num_samples)

    # Format data for LLM training
    formatted_samples = []
    # for audio_id, cur_text, emotion,  in sampled_data:
    for cur_audio_id, cur_text in sampled_data:
        # convert audio to units
        units = s2u(f"styletalk/audio/{cur_audio_id}")
        sample = {
            "prefix": f"Transcribe following speech input: {units} > ",
            "plain_text": f"{cur_text.strip()}",
        }
        formatted_samples.append(sample)

    return formatted_samples

def extract_samples_librispeech(data_dir, s2u, num_samples=100):
    """
    Extract audio and transcript samples from LibriSpeech dataset for ASR task.

    Parameters:
    data_dir (str): Path to LibriSpeech dataset directory (e.g., 'train-clean-100/')
    s2u: Speech2Unit model instance
    num_samples (int): Number of samples to extract

    Returns:
    list: List of dictionaries containing formatted prompts and responses
    """
    import os
    from pathlib import Path

    # Store all audio-transcript pairs
    all_samples = []

    # Walk through the dataset directory
    for speaker_id in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_id)
        if not os.path.isdir(speaker_path):
            continue

        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue

            # Read transcript file
            trans_file = os.path.join(
                chapter_path, f"{speaker_id}-{chapter_id}.trans.txt"
            )
            if not os.path.exists(trans_file):
                continue

            # Create mapping of utterance_id to transcript
            transcripts = {}
            with open(trans_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        file_id, text = parts
                        text = text.lower()
                        transcripts[file_id] = text

            # Match audio files with transcripts
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

    # Randomly sample if we have more data than requested
    if num_samples < len(all_samples):
        all_samples = random.sample(all_samples, num_samples)

    # Format samples for LLM training
    formatted_samples = []
    for sample in all_samples:
        # Convert audio to units using the provided speech2unit model
        units = s2u(sample["audio_path"])
        formatted_sample = {
            "prefix": f"Transcribe following speech input: {units} ",
            "plain_text": f"{sample['text'].strip()}",
        }
        formatted_samples.append(formatted_sample)

    return formatted_samples


def save_to_jsonl(samples, output_file):
    """
    Save formatted samples to JSONL file

    Parameters:
    samples (list): List of dictionaries containing formatted samples
    output_file (str): Path to output JSONL file
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + "\n")


# Usage example
if __name__ == "__main__":
    # Path to LibriSpeech dataset
    librispeech_dir = "/shared/NAS_SSD/wooyeol/LibriSpeech/train-clean-100/train-clean-100/"

    # Your CSV data string would go here
    # csv_data_path = "styletalk/train.csv"
    # csv_data_path = "styletalk/eval_without_weather_465.csv"

    ckpt_dir = "/home/jhwan98/EmoSDS/SpeechGPT/speechgpt/utils/speech2unit/"
    s2u = Speech2UnitCustom(ckpt_dir=ckpt_dir)

    # Extract and format samples
    # samples = extract_samples(csv_data_path, s2u, num_samples=20000)
    samples = extract_samples_librispeech(librispeech_dir, s2u, num_samples=50000)

    # Save to JSONL file
    # output_file = "stage2/asr_data_eval.jsonl"
    # output_file = "stage2/asr_data_train.jsonl"
    output_file = "stage2/asr_data_librispeech_train.jsonl"
    save_to_jsonl(samples, output_file)

    # Print example of formatted data
    print("\nExample of formatted data:")
    print(json.dumps(samples[0], indent=2))
    print(f"\nTotal samples saved to {output_file}: {len(samples)}")
