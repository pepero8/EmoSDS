import typer
from typing_extensions import Annotated
from collections import defaultdict


def asr_ser_split_data(jsonl_path, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    import json
    import random
    import math

    random.seed(42)

    train_data, valid_data, test_data = [], [], []

    samples_per_emo = defaultdict(list)

    # max_residual_length = 0  # 임시
    # invalid_res_emos = []  # 임시

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            plain_text = data["plain_text"]
            emotion = plain_text.split()[0].strip("<>")

            after_answer = plain_text.split("EmoSDS:")[1].strip()
            res_emotion = after_answer[
                after_answer.find("<") + 1 : after_answer.find(">")
            ].strip()

            # 임시
            # if res_emotion not in [
            #     "cheerful",
            #     "friendly",
            #     "unfriendly",
            #     "neutral",
            #     "sad",
            # ]:
            # invalid_res_emos.append(res_emotion)  # 임시

            if emotion != "fear" and emotion != "disgust":
                # samples_per_emo[emotion].append(data)
                samples_per_emo[res_emotion].append(data)

            # 임시
            # if max_residual_length < data["residual_length"]:
            #     max_residual_length = data["residual_length"]

    # print(f"max_residual_length: {max_residual_length}")
    # print(f"invalid emos: {invalid_res_emos}")  # 임시

    # exit()

    # > Print emotion statistics
    print("\nASR+SER dataset emotion Statistics:")
    for emotion, sample_list in sorted(samples_per_emo.items()):
        print(f"    {emotion}: {len(sample_list)} samples")

    exit()

    def extract_balanced_samples(samples_per_emo, total_samples, thres):
        num_samples_per_emotion = total_samples // len(samples_per_emo)
        remaining_samples = total_samples % len(samples_per_emo)

        selected_samples = []
        for emotion in samples_per_emo:
            n_samples = num_samples_per_emotion + (1 if remaining_samples > 0 else 0)
            if remaining_samples > 0:
                remaining_samples -= 1

            if len(samples_per_emo[emotion]) >= n_samples:
                selected = random.sample(samples_per_emo[emotion], n_samples)
                for sample in selected:
                    samples_per_emo[emotion].remove(sample)
                selected_samples.extend(selected)
            else:
                # > if we don't have enough, leave #thres samples and take rest
                selected = random.sample(
                    samples_per_emo[emotion], len(samples_per_emo[emotion]) - thres
                )
                for sample in selected:
                    samples_per_emo[emotion].remove(sample)
                selected_samples.extend(selected)
                # selected_samples.extend(samples_per_emo[emotion])
                # samples_per_emo[emotion] = []

        return selected_samples

    test_samples = extract_balanced_samples(samples_per_emo, 45, 9)
    valid_samples = extract_balanced_samples(samples_per_emo, 45, 0)
    train_samples = extract_balanced_samples(samples_per_emo, 2000, 0)

    random.shuffle(test_samples)
    random.shuffle(valid_samples)
    random.shuffle(train_samples)

    return train_samples, valid_samples, test_samples


def asr_ser_split_data_splitted_task(
    jsonl_path, asr_valid_samples, asr_test_samples, asr_train_samples=None
):
    import json
    import random
    import math

    random.seed(42)

    train_data, valid_data, test_data = [], [], []

    samples_per_emo = defaultdict(list)
    asr_samples = []

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            task = data["task"]
            plain_text = data["plain_text"]

            if task == "asr":
                text = plain_text.strip()
                asr_samples.append(data)
            elif task == "ser":
                emotion = plain_text.strip("<>")
                samples_per_emo[emotion].append(data)

    # > Print emotion statistics
    print("\nASR+SER dataset emotion Statistics:")
    for emotion, sample_list in sorted(samples_per_emo.items()):
        print(f"    {emotion}: {len(sample_list)} samples")

    def extract_balanced_samples(samples_per_emo, total_samples, thres):
        num_samples_per_emotion = total_samples // len(samples_per_emo)
        remaining_samples = total_samples % len(samples_per_emo)

        selected_samples = []
        for emotion in samples_per_emo:
            n_samples = num_samples_per_emotion + (1 if remaining_samples > 0 else 0)
            if remaining_samples > 0:
                remaining_samples -= 1

            if len(samples_per_emo[emotion]) >= n_samples:
                selected = random.sample(samples_per_emo[emotion], n_samples)
                for sample in selected:
                    samples_per_emo[emotion].remove(sample)
                selected_samples.extend(selected)
            else:
                # > if we don't have enough, leave #thres samples and take rest
                selected = random.sample(
                    samples_per_emo[emotion], len(samples_per_emo[emotion]) - thres
                )
                for sample in selected:
                    samples_per_emo[emotion].remove(sample)
                selected_samples.extend(selected)
                # selected_samples.extend(samples_per_emo[emotion])
                # samples_per_emo[emotion] = []

        return selected_samples

    test_samples_ser = extract_balanced_samples(samples_per_emo, 250, 0)
    valid_samples_ser = extract_balanced_samples(samples_per_emo, 250, 100)
    train_samples_ser = extract_balanced_samples(samples_per_emo, 20000, 0)

    # Split ASR samples
    random.shuffle(asr_samples)
    test_samples_asr = asr_samples[:asr_test_samples]
    valid_samples_asr = asr_samples[
        asr_test_samples : asr_test_samples + asr_valid_samples
    ]
    train_samples_asr = asr_samples[
        asr_test_samples
        + asr_valid_samples : asr_test_samples
        + asr_valid_samples
        + asr_train_samples
    ]

    # Merge and shuffle splits
    train_samples = train_samples_ser + train_samples_asr
    valid_samples = valid_samples_ser + valid_samples_asr
    test_samples = test_samples_ser + test_samples_asr

    random.shuffle(test_samples)
    random.shuffle(valid_samples)
    random.shuffle(train_samples)

    return train_samples, valid_samples, test_samples


def check_split_distribution(split_data, split_name):
    split_emotions = defaultdict(int)
    for sample in split_data:
        emotion = sample["plain_text"].split()[0].strip("<>")
        split_emotions[emotion] += 1

    print(f"\n{split_name} Split Distribution:")
    print("-" * 30)
    for emotion, count in split_emotions.items():
        print(f"{emotion}: {count} samples")
    print(f"Total: {len(split_data)} samples")


def check_split_distribution_splitted_task(split_data, split_name):
    split_emotions = defaultdict(int)
    for sample in split_data:
        if sample["task"] == "ser":
            emotion = sample["plain_text"].strip("<>")
            split_emotions[emotion] += 1

    print(f"\n{split_name} Split Distribution:")
    print("-" * 30)
    for emotion, count in split_emotions.items():
        print(f"{emotion}: {count} samples")
    print(f"Total: {len(split_data)} samples")


def save_to_jsonl(samples, output_file):
    import json

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + "\n")


def main(
    data: Annotated[
        str,
        typer.Argument(help="json or jsonl data path to perform split"),
    ],
    out_dir: Annotated[
        str,
        typer.Argument(help="output path"),
    ],
):

    # train, valid, test = asr_ser_split_data(jsonl_path=data)
    asr_ser_split_data(jsonl_path=data)
    # train, valid, test = asr_ser_split_data_splitted_task(
    #     jsonl_path=data,
    #     asr_valid_samples=250,
    #     asr_test_samples=250,
    #     asr_train_samples=20000,
    # )

    # check_split_distribution(train, "Train")
    # check_split_distribution(valid, "Validation")
    # check_split_distribution(test, "Test")
    # # check_split_distribution_splitted_task(train, "Train")
    # # check_split_distribution_splitted_task(valid, "Validation")
    # # check_split_distribution_splitted_task(test, "Test")

    # # output_path_train = f"{out_dir}/asr_ser_task_train_balanced_splitted_task.jsonl"
    # # output_path_test = f"{out_dir}/asr_ser_task_test_balanced_splitted_task.jsonl"
    # # output_path_valid = f"{out_dir}/asr_ser_task_valid_balanced_splitted_task.jsonl"
    # output_path_train = (
    #     f"{out_dir}/asr_ser_task_train_balanced_dailytalk_newresidual.jsonl"
    # )
    # output_path_test = f"{out_dir}/asr_ser_task_test_balanced_dailytalk_newresidual.jsonl"
    # output_path_valid = (
    #     f"{out_dir}/asr_ser_task_valid_balanced_dailytalk_newresidual.jsonl"
    # )

    # save_to_jsonl(train, output_path_train)
    # save_to_jsonl(test, output_path_test)
    # save_to_jsonl(valid, output_path_valid)

    # print(f"\nTotal train samples saved to {output_path_train}: {len(train)}")
    # print(f"\nTotal test samples saved to {output_path_test}: {len(test)}")
    # print(f"\nTotal valid samples saved to {output_path_valid}: {len(valid)}")


if __name__ == "__main__":
    typer.run(main)

# python3 utils/asr_ser_split_jsonl_dataset.py /home/jhwan98/EmoSDS/data/asr_ser/layer6_k1000/merged/asr_ser_task_dailytalk_newresidual.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer6_k1000/merged
# python3 utils/asr_ser_split_jsonl_dataset.py /home/jhwan98/EmoSDS/data/unified/layer6_k1000_merged/unified_task_styletalk_train_balanced.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer6_k1000/merged
# python3 utils/asr_ser_split_jsonl_dataset.py /home/jhwan98/EmoSDS/data/unified/layer6_k1000_merged/unified_task_dailytalk_test_residual.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer6_k1000/merged
# python3 utils/asr_ser_split_jsonl_dataset.py /home/jhwan98/EmoSDS/data/unified/layer6_k1000_merged/unified_task_dailytalk_test_balanced_newresidual_2000_no883.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer6_k1000/merged
# python3 utils/asr_ser_split_jsonl_dataset.py /home/jhwan98/EmoSDS/data/unified/layer6_k1000_merged/unified_task_styletalk_test_balanced.jsonl /home/jhwan98/EmoSDS/data/asr_ser/layer6_k1000/merged
