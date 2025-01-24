def verify_response(response):
    valid_emotions = {
        "anger",
        "disgust",
        "fear",
        "happiness",
        "neutral",
        "sadness",
        "surprise",
    }

    required_turns = [
        "history_turns",
        "current_turn1",
        "current_turn2",
        "response_turn1",
        "response_turn2",
    ]
    for turn in required_turns:
        if turn not in response:
            # raise ValueError(f"Missing required turn: {turn}")
            print(f"Missing required turn: {turn}")
            return False

    # > Validate num of history turns
    history_turns = response["history_turns"]
    if len(history_turns) != 3:
        # raise ValueError("History must contain exactly 3 turns")
        print(f"History must contain exactly 3 turns")
        return False

    # > Check turn alternation (must be B-A-B)
    speakers = [turn.split(":")[0].strip() for turn in history_turns]
    if not (speakers[0] == "B" and speakers[1] == "A" and speakers[2] == "B"):
        # raise ValueError("History turns must follow B-A-B pattern")
        print(f"History turns must follow B-A-B pattern")
        return False

    # > Validate emotions
    all_turns = history_turns + [
        response["current_turn1"],
        response["current_turn2"],
        response["response_turn1"],
        response["response_turn2"],
    ]

    for turn in all_turns:
        try:
            emotion = turn.split("<")[1].split(">")[0].strip()
        except IndexError:
            # raise ValueError(f"Invalid emotion format in turn: {turn}")
            print(f"Invalid emotion format in turn: {turn}")
            return False

        # Check emotion validity
        if emotion not in valid_emotions:
            # raise ValueError(f"Invalid emotion: {emotion}")
            print(f"Invalid emotion: {emotion}")
            return False

        # Check for 'A' or 'B' mentions
        # speaker = turn.split(":")[0].strip()
        # text = turn.split(">")[1].strip()
        # if "A" in text or "B" in text:
        #     raise ValueError(f"Dialogue text contains speaker reference: {turn}")

    # If all checks pass, return True (optional, but can be useful)
    return True


def synthesize_ravdess(data_dir):
    from collections import defaultdict
    from pathlib import Path
    import json
    from json import JSONDecodeError

    map_emo = {
        "01": "neutral",
        # "02": "calm",
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

    # >> Synthesize history and responses for each emotion and text combination
    for text_idx, text in map_text.items():
        for emo_idx, emotion in map_emo.items():
            response_str = call_gpt(sample=f"A: <{emotion}> {text}")
            try:
                response = json.loads(
                    # response_str.strip("```").split("json")[-1].strip()
                    response_str.strip()
                )
            except JSONDecodeError:
                print(
                    f"[{emotion}, {text}] Not a json format: [{response_str.strip()}]"
                )
                continue
            if not verify_response(response):
                print(
                    f"This response for [{emotion}, {text}] is invalid: {response_str.strip()}"
                )
                continue

            all_samples.append(response)

    return all_samples


def synthesize_esd(file_path):
    import json
    from json import JSONDecodeError

    all_samples = []
    # extracted_texts = []

    emotions = {
        "neutral",
        "happiness",
        "sadness",
        "anger",
        "fear",
        "disgust",
        "surprise",
    }

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                text = parts[1]
                for emotion in emotions:
                    response_str = call_gpt(sample=f"A: <{emotion}> {text}")
                    try:
                        response = json.loads(
                            # response_str.strip("```").split("json")[-1].strip()
                            response_str.strip()
                        )
                    except JSONDecodeError:
                        print(
                            f"[{emotion}, {text}] Not a json format: [{response_str.strip()}]"
                        )
                        continue
                    if not verify_response(response):
                        print(
                            f"This response for [{emotion}, {text}] is invalid: {response_str.strip()}"
                        )
                        continue

                    all_samples.append(response)

                # extracted_texts.append(parts[1])

    return all_samples


def call_gpt(sample):
    import openai
    import os

    os.environ["OPENAI_API_KEY"] = (
        "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"
    )

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an human-like dialogue data expert that imitates the real human-to-human spoken dialogue.
The speaking style should be very natural in the dialogue context.
Important: Consider a scenario that after the history turns, there is a current turn with text and emotion.
The same text can have different emotions. The different current emotions would make the response text fairly different in terms of semantics.
Just one sentence for each turn. The sentence is spoken spontaneously and not too formal.
[Rules you must follow]:
0. We use special token <> to represent the emotion type that you have to generate.
1. Important: do not use other class that is not defined below!!!
1.1 emotions: anger, disgust, fear, happiness, neutral, sadness, surprise
Don’t use other emotions!
2. Use diverse emotions in the conversation context.
3. The text of current turn is read in specified emotion, and the response turn should carefully consider the current turn, response naturally, not just copying current emotion.
4. There are two speakers (A and B) in the dialogue. Two speakers talk with back and forth interaction.
5. Speaker A or B must not be mentioned in any dialogue text.
6. Each turn should follow the format: <speaker>: <emotion> <text>
7. The order of turns is history turns -> current turn -> response turn.
8. The transition of dialogue turns should be very consistent and the conversation follows the common sense.
9. The dialouge contains emotional variation.
10. The output valid dictionary format is as below:
{
"history_turns": [ "<speaker>: <emotion> <text>", ...], # 3 history turns
"current_turn1": "<speaker>: <emotion1> <text>", # the word of current turn is uttered with specified emotion1
"current_turn2": "<speaker>: <emotion2> <text>", # the word of current turn is uttered with specified emotion2
"response_turn1": "<speaker>: <emotion1> <text1>", # emotion and response to current_turn1
"response_turn2": "<speaker>: <emotion2> <text2>" # emotion and response to current_turn2
}
11. Output the valid dictionary example, so that it can be parsed as dictionary.
12. For <speaker>, only use A or B.""",
            },
            {
                "role": "user",
                "content": f"""Given the context of the current text and its associated emotion, generate three prior conversational turns as history to provide context. Then:
1. Predict the appropriate emotion and response text based on the given history and current emotion and text.
2. Generate an alternative emotion for the current text that conveys a different nuance, while keeping the text itself unchanged.
3. Based on this alternative emotion, generate a new appropriate emotion and response text.
The starting speaker of history is B.
Feel free to imagine the dialogue content but it should be based on common sense. We use <emotion> to represent emotion.
The purpose of this process is to demonstrate that even when the current text remains the same, its meaning and intention can shift based on the emotion behind it, leading to different responses. The two sets of responses should reflect the distinct emotional contexts.

prompt:
{sample}""",
            },
        ],
    )

    # print("Num of responses: " + str(len(response.choices)))
    # print("Response: " + response.choices[0].message.content)

    return response.choices[0].message.content


import typer
from typing_extensions import Annotated


def save_to_jsonl(samples, output_file):
    import json

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + "\n")


def main(
    ravdess_dir: Annotated[
        str, typer.Option(help="Directory path to ravdess dataset")
    ] = None,
    esd_path: Annotated[str, typer.Option(help="Directory path to ESD dataset")] = None,
):
    # samples = synthesize_ravdess(ravdess_dir)
    # output_path = (
    #     "/home/jhwan98/EmoSDS/data/synthesized/unified/ravdess_without_audio.jsonl"
    # )
    # save_to_jsonl(samples=samples, output_file=output_path)
    samples = synthesize_esd(esd_path)
    output_path = (
        "/home/jhwan98/EmoSDS/data/synthesized/unified/esd_without_audio.jsonl"
    )
    save_to_jsonl(samples=samples, output_file=output_path)


if __name__ == "__main__":
    typer.run(main)

# python3 utils/make_synthesized_dataset.py --ravdess-dir /shared/NAS_SSD/jhl/futureinternet/ravdess_audio
# python3 utils/make_synthesized_dataset.py --esd-path /home/jhwan98/EmoSDS/data/synthesized/unified/esd_texts.txt
