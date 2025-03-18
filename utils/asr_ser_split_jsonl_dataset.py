import typer
from typing_extensions import Annotated
from collections import defaultdict


def asr_ser_split_data(jsonl_path):
    import json
    import random

    random.seed(42)

    samples_per_emo = defaultdict(list)

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            plain_text = data["plain_text"]
            emotion = plain_text.split()[0].strip("<>")
            samples_per_emo[emotion].append(data)

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
                selected = random.sample(
                    samples_per_emo[emotion], len(samples_per_emo[emotion]) - thres
                )
                for sample in selected:
                    samples_per_emo[emotion].remove(sample)
                selected_samples.extend(selected)

        return selected_samples

    test_samples = extract_balanced_samples(samples_per_emo, 100, 9)
    valid_samples = extract_balanced_samples(samples_per_emo, 100, 0)
    train_samples = extract_balanced_samples(samples_per_emo, 20000, 0)

    random.shuffle(test_samples)
    random.shuffle(valid_samples)
    random.shuffle(train_samples)

    return train_samples, valid_samples, test_samples


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
    output_path_train = (
        f"{out_dir}/asr_ser_task_esd_train.jsonl"
    )
    output_path_test = f"{out_dir}/asr_ser_task_esd_test.jsonl"
    output_path_valid = (
        f"{out_dir}/asr_ser_task_esd_valid.jsonl"
    )

    save_to_jsonl(train, output_path_train)
    save_to_jsonl(test, output_path_test)
    save_to_jsonl(valid, output_path_valid)


if __name__ == "__main__":
    typer.run(main)
