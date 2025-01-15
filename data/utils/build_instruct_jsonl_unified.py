import pandas as pd
import random
import json
import sys
sys.path.append("/home/jhwan98/EmoSDS/SpeechGPT/speechgpt")
from utils.speech2unit.speech2unit import Speech2UnitCustom


# def create_style_mapping():
#     """
#     Create a mapping dictionary from (emotion, speed, volume) combinations to styles

#     Returns:
#     dict: Dictionary with (emotion, speed, volume) tuple keys and style string values
#     """

#     # Define the mapping based on the provided combinations
#     mappings = {
#         ("neutral", "slow", "quiet"): "sleepy",
#         ("neutral", "slow", "normal"): "bored",
#         ("neutral", "slow", "loud"): "solemn",
#         ("neutral", "normal", "quiet"): "whisper",
#         ("neutral", "normal", "normal"): "neutral",
#         ("neutral", "normal", "loud"): "projected",
#         ("neutral", "fast", "quiet"): "hesitant",
#         ("neutral", "fast", "normal"): "interested",
#         ("neutral", "fast", "loud"): "urgent",
#         ("angry", "slow", "quiet"): "simmering",
#         ("angry", "slow", "normal"): "resentful",
#         ("angry", "slow", "loud"): "menacing",
#         ("angry", "normal", "quiet"): "sarcastic",
#         ("angry", "normal", "normal"): "angry",
#         ("angry", "normal", "loud"): "shouting",
#         ("angry", "fast", "quiet"): "seething",
#         ("angry", "fast", "normal"): "furious",
#         ("angry", "fast", "loud"): "raging",
#         ("cheerful", "slow", "quiet"): "calm",
#         ("cheerful", "slow", "normal"): "gentle",
#         ("cheerful", "slow", "loud"): "heartfelt",
#         ("cheerful", "normal", "quiet"): "warm",
#         ("cheerful", "normal", "normal"): "happy",
#         ("cheerful", "normal", "loud"): "excited",
#         ("cheerful", "fast", "quiet"): "giggling",
#         ("cheerful", "fast", "normal"): "laughing",
#         ("cheerful", "fast", "loud"): "joyful",
#         ("sad", "slow", "quiet"): "depressed",
#         ("sad", "slow", "normal"): "sorrowful",
#         ("sad", "slow", "loud"): "lamenting",
#         ("sad", "normal", "quiet"): "tearful",
#         ("sad", "normal", "normal"): "sad",
#         ("sad", "normal", "loud"): "crying",
#         ("sad", "fast", "quiet"): "anxious",
#         ("sad", "fast", "normal"): "distressed",
#         ("sad", "fast", "loud"): "wailing",
#         ("excited", "slow", "quiet"): "anticipatory",
#         ("excited", "slow", "normal"): "eager",
#         ("excited", "slow", "loud"): "building",
#         ("excited", "normal", "quiet"): "intrigued",
#         ("excited", "normal", "normal"): "excited",
#         ("excited", "normal", "loud"): "enthusiastic",
#         ("excited", "fast", "quiet"): "ecstatic",
#         ("excited", "fast", "normal"): "thrilled",
#         ("excited", "fast", "loud"): "overjoyed",
#         ("friendly", "slow", "quiet"): "kind",
#         ("friendly", "slow", "normal"): "welcoming",
#         ("friendly", "slow", "loud"): "reassuring",
#         ("friendly", "normal", "quiet"): "soft-spoken",
#         ("friendly", "normal", "normal"): "friendly",
#         ("friendly", "normal", "loud"): "encouraging",
#         ("friendly", "fast", "quiet"): "playful",
#         ("friendly", "fast", "normal"): "warm-hearted",
#         ("friendly", "fast", "loud"): "upbeat",
#         ("terrified", "slow", "quiet"): "trembling",
#         ("terrified", "slow", "normal"): "uneasy",
#         ("terrified", "slow", "loud"): "fearful",
#         ("terrified", "normal", "quiet"): "nervous",
#         ("terrified", "normal", "normal"): "afraid",
#         ("terrified", "normal", "loud"): "panicked",
#         ("terrified", "fast", "quiet"): "frantic",
#         ("terrified", "fast", "normal"): "hysterical",
#         ("terrified", "fast", "loud"): "screaming",
#         ("shouting", "slow", "quiet"): "muttering",
#         ("shouting", "slow", "normal"): "assertive",
#         ("shouting", "slow", "loud"): "booming",
#         ("shouting", "normal", "quiet"): "speaking-up",
#         ("shouting", "normal", "normal"): "projected",
#         ("shouting", "normal", "loud"): "yelling",
#         ("shouting", "fast", "quiet"): "blurting",
#         ("shouting", "fast", "normal"): "hollering",
#         ("shouting", "fast", "loud"): "roaring",
#         ("unfriendly", "slow", "quiet"): "cold",
#         ("unfriendly", "slow", "normal"): "weary",
#         ("unfriendly", "slow", "loud"): "grumbling",
#         ("unfriendly", "normal", "quiet"): "sarcastic",
#         ("unfriendly", "normal", "normal"): "disgusted",
#         ("unfriendly", "normal", "loud"): "mocking",
#         ("unfriendly", "fast", "quiet"): "sneering",
#         ("unfriendly", "fast", "normal"): "spiteful",
#         ("unfriendly", "fast", "loud"): "hostile",
#         ("whispering", "slow", "quiet"): "hush",
#         ("whispering", "slow", "normal"): "gentle-whisper",
#         ("whispering", "slow", "loud"): "muffled",
#         ("whispering", "normal", "quiet"): "whisper",
#         ("whispering", "normal", "normal"): "low-key",
#         ("whispering", "normal", "loud"): "subdued",
#         ("whispering", "fast", "quiet"): "quick-hush",
#         ("whispering", "fast", "normal"): "secretive",
#         ("whispering", "fast", "loud"): "breathy",
#         ("hopeful", "slow", "quiet"): "dreaming",
#         ("hopeful", "slow", "normal"): "optimistic",
#         ("hopeful", "slow", "loud"): "longing",
#         ("hopeful", "normal", "quiet"): "gentle-hope",
#         ("hopeful", "normal", "normal"): "hopeful",
#         ("hopeful", "normal", "loud"): "inspired",
#         ("hopeful", "fast", "quiet"): "yearning",
#         ("hopeful", "fast", "normal"): "motivated",
#         ("hopeful", "fast", "loud"): "driven",
#     }

#     return mappings

def extract_samples_dailytalk(data_path, s2u, num_samples=100, test_size=0.1):
    import os
    from collections import defaultdict
    import json

    # Load metadata
    with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    # Store all valid conversation pairs
    formatted_samples = []
    # Track emotion statistics
    emotion_stats = defaultdict(int)

    for dialog_idx in metadata.keys():
        dialog_data = metadata[dialog_idx]
        dialog_turns = len(dialog_data)

        dialog_history = {}
        # Process consecutive turns to create current-response pairs
        for turn_idx in range(dialog_turns - 1):
            curr_turn = dialog_data[str(turn_idx)]
            next_turn = dialog_data[str(turn_idx + 1)]

            # Update history
            if turn_idx > 0:
                dialog_history[str(turn_idx - 1)] = dialog_data[str(turn_idx - 1)]['text']
                if len(dialog_history) > 5:
                    dialog_history.pop(next(iter(dialog_history))) # removes the first element

            # Get current and response emotions
            curr_emotion = curr_turn['emotion']
            resp_emotion = next_turn['emotion']

            # Convert "no emotion" to "neutral" for consistency
            curr_emotion = "neutral" if curr_emotion == "no emotion" else curr_emotion
            resp_emotion = "neutral" if resp_emotion == "no emotion" else resp_emotion

            # Update emotion statistics
            emotion_stats[curr_emotion] += 1

            # Get audio path
            curr_audio_path = os.path.join(
                data_path, 
                'data', 
                str(dialog_idx),
                f"{turn_idx}_{curr_turn['speaker']}_d{dialog_idx}.wav"
            )

            # Build history string
            history = ""
            if turn_idx > 0:
                spk_names = ("Answer", "Input")
                spk_idx = 0
                for turn, text in reversed(dialog_history.items()):
                    history = f"{spk_names[spk_idx]}: {text}\n" + history
                    spk_idx = int(not spk_idx)

            # Convert audio to units if file exists
            if os.path.exists(curr_audio_path):
                units = s2u(curr_audio_path)

                sample = {
                    "prefix": f'''
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

''' + \
f"{history} " + \
f"Input: {units} ",
                    "plain_text": f"<{curr_emotion}> {curr_turn['text'].strip()} Answer: <{resp_emotion}> {next_turn['text'].strip()}",
                }

                formatted_samples.append(sample)

    # Print emotion statistics
    print("\nEmotion Statistics:")
    total_samples = sum(emotion_stats.values())
    for emotion, count in sorted(emotion_stats.items()):
        percentage = (count / total_samples) * 100
        print(f"{emotion}: {count} samples ({percentage:.2f}%)")

    # If requested samples are more than available data, use all data
    if num_samples > len(formatted_samples):
        # return formatted_samples
        pass
    else:
        # Randomly sample the data
        # return random.sample(formatted_samples, num_samples)
        formatted_samples = random.sample(formatted_samples, num_samples)

    # Calculate number of samples for test set
    num_test = int(num_samples * test_size)
    num_train = num_samples - num_test
    
    # Shuffle and split the samples
    random.shuffle(formatted_samples)
    train_samples = formatted_samples[:num_train]
    test_samples = formatted_samples[num_train:]

    return train_samples, test_samples
    

def extract_samples(csv_data_path, s2u, num_samples=100):
    """
    Extract audio IDs, styles, response texts from CSV data and format for LLM training

    Parameters:
    csv_data (str): CSV content path
    num_samples (int): Number of samples to extract

    Returns:
    list: List of dictionaries containing formatted prompts and responses
    """
    from collections import defaultdict
    # Read CSV data
    # df = pd.read_csv(pd.StringIO(csv_data))
    df = pd.read_csv(csv_data_path)

    # Get style mapping
    # style_mapping = create_style_mapping()

    # Extract required columns and create style combinations
    audio_style_data = []
    emotion_stats = defaultdict(int)
    for _, row in df.iterrows():
        # curr_combo = row["curr_emotion"]
        # res_combo = (row["res_emotion"], row["res_speed"], row["res_volume"])
        # curr_style = style_mapping.get(
        #     curr_combo, "neutral"
        # )  # Default to neutral if combination not found
        # res_style = style_mapping.get(res_combo, "neutral")
        try:
            if row["curr_text"].split(": ")[0] == 'B':
                prev_1 = row["context"].split("B :")[0].split("A :")[1].strip(), # prev_A1
                prev_2 = row["context"].split("B :")[1].split("A :")[0].strip(), # prev_B
                prev_3 = row["context"].split("B :")[1].split("A :")[1].strip(), # prev_A2
            elif row["curr_text"].split(": ")[0] == 'A':
                prev_1 = row["context"].split("A :")[0].split("B :")[1].strip(), # prev_B1
                prev_2 = row["context"].split("A :")[1].split("B :")[0].strip(), # prev_A
                prev_3 = row["context"].split("A :")[1].split("B :")[1].strip(), # prev_B2
        except:
            raise Exception(f"context: {row['context']}")

        emotion_stats[row["curr_emotion"]] += 1
        
        audio_style_data.append(
            (
                prev_1[0],
                prev_2[0],
                prev_3[0],
                row["curr_audio_id"],
                # row["curr_text"],
                row["curr_text"].replace("B:", "", 1).strip(), # remove 'B:' part
                row["curr_emotion"],
                row["res_text"],
                row["res_emotion"],
            )
        )

    # Print emotion statistics
    print("\nEmotion Statistics:")
    total_samples = sum(emotion_stats.values())
    for emotion, count in sorted(emotion_stats.items()):
        percentage = (count / total_samples) * 100
        print(f"{emotion}: {count} samples ({percentage:.2f}%)")

    # If requested samples are more than available data, use all data
    if num_samples > len(audio_style_data):
        sampled_data = audio_style_data
    else:
        # Randomly sample the data
        sampled_data = random.sample(audio_style_data, num_samples)

    # Format data for LLM training
    formatted_samples = []
    # for audio_id, cur_text, emotion,  in sampled_data:
    for prev_1, prev_2, prev_3, cur_audio_id, cur_text, cur_style, resp_text, resp_style in sampled_data:
        # convert audio to units
        units = s2u(f"styletalk/audio/{cur_audio_id}")
        sample = {
            "prefix": f'''
# Task
From now on, you are an intelligent voice assistant. You need to provide useful, consistent to the dialogue context, emotionally approval natural response to the user's input speech.
Given user speech and history, you need to transcribe the user speech, identify the speaking style, predict appropriate response style, and predict appropriate response text according to the context.
The speaking style should be one of following 11 styles: neutral, angry, cheerful, sad, excited, friendly, terrified, shouting, unfriendly, whispering, hopeful

# Examples
Following examples show example responses to the transcribed input speech with speaking style and history. The caption in angle brackets indicate speaking style of the transcription."

## Example 1
Input: It just tastes so good, you know?
Answer: That's awesome! It's always great to find something you enjoy. Do you use it on anything specific, like toast or cooking?
Input: <excited> I can't believe it's not butter!
Answer: <friendly> Oh wow, you're really passionate about this! So, what is it about "I Can't Believe It's Not Butter" that's got you so excited?

## Example 2
Input: <angry> I can't believe it's not butter!
Answer: <neutral> Whoa, okay, let's take a deep breath and try to calm down. Are you actually upset that it's not butter? What's really going on here?

## Example 3
Input: I watched a baseball game on the weekend
Answer: Oh cool! how was it?
Input: <sad> The game wasn't bad
Answer: <sad> You don't seem too happy, did your team lose?

## Example 4
Input: I watched a baseball game on the weekend
Answer: Oh cool! how was it?
Input: <excited> The game wasn't bad
Answer: <friendly> That's great to hear! Did your favorite team win?

Here's the prompt:

''' + \
f"Answer: {prev_1} Input: {prev_2} Answer: {prev_3} " + \
f"Input: {units} ",
            # "prefix": f"Identify speaking style of given speech: {units}. Provide only the style label without any explanation. >",
            # 만약 [Humans], [SpeechGPT]와 같은 특수 토큰을 포함한 대화 '형식'을 학습하도록 하고 싶다면 이를 plain_text에 넣을 것.
            # "plain_text": f"{cur_style}] {cur_text.strip()} Answer: [{resp_style}] {resp_text.strip()}",
            "plain_text": f"<{cur_style}> {cur_text.strip()} Answer: <{resp_style}> {resp_text.strip()}",
        }
        formatted_samples.append(sample)

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
    # Your CSV data string would go here
    # csv_data_path = "styletalk/train.csv"
    # csv_data_path = "styletalk/eval_without_weather_465.csv"

    dailytalk_path = "/shared/NAS_SSD/jhl/futureinternet/dailytalk"

    ckpt_dir = "/home/jhwan98/EmoSDS/SpeechGPT/speechgpt/utils/speech2unit/"
    s2u = Speech2UnitCustom(ckpt_dir=ckpt_dir)

    # Extract and format samples
    # samples = extract_samples(csv_data_path, s2u, num_samples=20000)
    train_samples, test_samples = extract_samples_dailytalk(dailytalk_path, s2u, num_samples=200000, test_size=0.05)

    # Save to JSONL file
    # output_file = "stage3/dialogue_prediction_unified_data_with_history_eval.jsonl"
    # output_file = "stage3/dialogue_prediction_unified_data_eval_style_modified.jsonl"
    output_file = "stage3/dialogue_prediction_unified_data_dailytalk_history_train_eval.jsonl"
    test_file = "stage3/dialogue_prediction_unified_data_dailytalk_history_test.jsonl"
    # save_to_jsonl(samples, output_file)
    save_to_jsonl(train_samples, output_file)
    save_to_jsonl(test_samples, test_file)

    # Print example of formatted data
    print("\nExample of formatted data:")
    print(json.dumps(train_samples[0], indent=2))
    print(json.dumps(test_samples[0], indent=2))
    print(f"\nTotal train samples saved to {output_file}: {len(train_samples)}")
    print(f"\nTotal test samples saved to {test_file}: {len(test_samples)}")
