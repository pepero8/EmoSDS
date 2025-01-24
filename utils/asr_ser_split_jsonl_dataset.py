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

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            plain_text = data["plain_text"]
            emotion = plain_text.split()[0].strip("<>")

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

    test_samples = extract_balanced_samples(samples_per_emo, 50, 0)
    valid_samples = extract_balanced_samples(samples_per_emo, 50, 50)
    train_samples = extract_balanced_samples(samples_per_emo, 15000, 0)

    # # > Create balanced splits
    # for emotion, samples in samples_per_emo.items():
    #     shuffled_samples = samples.copy()
    #     random.shuffle(shuffled_samples)

    #     n_samples = len(shuffled_samples)
    #     n_train = int(n_samples * train_ratio)
    #     n_valid = int(n_samples * valid_ratio)

    #     train_data.extend(shuffled_samples[:n_train])
    #     valid_data.extend(shuffled_samples[n_train : n_train + n_valid])
    #     test_data.extend(shuffled_samples[n_train + n_valid :])

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

    train, valid, test = asr_ser_split_data(jsonl_path=data)

    check_split_distribution(train, "Train")
    check_split_distribution(valid, "Validation")
    check_split_distribution(test, "Test")

    output_path_train = f"{out_dir}/asr_ser_task_train_balanced.jsonl"
    output_path_test = f"{out_dir}/asr_ser_task_test_balanced.jsonl"
    output_path_valid = f"{out_dir}/asr_ser_task_valid_balanced.jsonl"

    save_to_jsonl(train, output_path_train)
    save_to_jsonl(test, output_path_test)
    save_to_jsonl(valid, output_path_valid)

    print(f"\nTotal train samples saved to {output_path_train}: {len(train)}")
    print(f"\nTotal test samples saved to {output_path_test}: {len(test)}")
    print(f"\nTotal valid samples saved to {output_path_valid}: {len(valid)}")


if __name__ == "__main__":
    typer.run(main)
