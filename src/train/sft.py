import logging
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import Trainer
from transformers import TrainingArguments
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    train_low_layers: bool = field(
        default=False,
        metadata={"help": ("only train low layers while training")},
    )


@dataclass
class DataArguments:
    data_path: str = field(default="", metadata={"help": "Path to the training data."})
    val_data_path: str = field(
        default=None, metadata={"help": "Path to the validation data"}
    )
    test_data_path: str = field(
        default=None, metadata={"help": "Path to the test data"}
    )
    test_data_path: str = field(
        default=None, metadata={"help": "Path to the test data"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    emo_token_list: str = field(
        default=None,
        metadata={
            "help": (
                "list of emotion tokens. Each emotion should be captured in angle bracket."
                "ex) '<anger> <disgust> <fear> <happiness>'"
            )
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the tokenized data"},
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    val_set_size: int = field(
        default=2000,
        metadata={"help": "val_set_size"},
    )
    preprocessing_num_workers: int = field(
        default=100,
        metadata={"help": "preprocessing_num_workers for tokenizing"},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "num_epochs"},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "learning_rate"},
    )
    output_dir: str = field(
        default="",
        metadata={"help": "output_dir"},
    )
    train_on_inputs: bool = field(
        default=True,
        metadata={"help": "if False, masks out inputs in loss"},
    )
    initial_global_step: int = field(
        default=0, metadata={"help": "initial_global_step"}
    )
    train_task: str = field(
        default=None, metadata={"help": "task to train. Options: ['asr', 'asr+ser', 'unified']"}
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    import torch
    from datasets import load_dataset
    from datasets import config
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        HfArgumentParser,
        DataCollatorForSeq2Seq,
    )
    from transformers.trainer_utils import get_last_checkpoint
    from utils.prompter import Prompter
    import sys
    import os
    from typing import Dict, List, Tuple

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set different cache directory
    config.HF_DATASETS_CACHE = training_args.cache_dir

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    prompter = Prompter()

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    ).to(torch.device(training_args.device))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def build_emo_dict(emo_tokens):
        dict = {}
        for i, emo in enumerate(emo_tokens):
            dict[emo.split(">")[0].split("<")[-1]] = i

        print(f"Emotion dictionary: {dict}")
        return dict

    # >> Extend vocab for speech units
    emo_tokens = []
    speaker_tokens = []
    valid_emotions = {}
    if training_args.train_task in ["unified", "asr+ser"]:
        emo_tokens = data_args.emo_token_list.split()
        valid_emotions = build_emo_dict(emo_tokens)
    if training_args.train_task == "unified":
        speaker_tokens = ["user: ", "EmoSDS: "]
    if "<sosp>" not in tokenizer.get_vocab():
        units_size = 1000
        logger.info(f"Add special unit tokens <0>-<{units_size-1}> to tokenizer.vocab")
        new_tokens = [f"<{x}>" for x in range(units_size)] + [
            "<sosp>",
            "<eosp>",
        ]
        tokenizer.add_tokens(new_tokens)
        if training_args.train_task in ["unified", "asr+ser"]:
            logger.info(f"Add emotion tokens to tokenizer.vocab")
            tokenizer.add_tokens(emo_tokens)
        if training_args.train_task == "unified":
            logger.info(f"Add speaker tokens to tokenizer.vocab")
            tokenizer.add_tokens(speaker_tokens)
    for token in ["<sosp>", "<eosp>"] + emo_tokens + speaker_tokens:
        if token not in tokenizer.get_vocab():
            logger.info(f"Add special unit tokens {token} to tokenizer.vocab")
            tokenizer.add_tokens([token])

    # >> resize embedding
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model_args.train_low_layers:
        logger.info("only update low layer parameters")
        for name, param in model.named_parameters():
            if (
                f"model.layers.0." in name
                or f"model.layers.1." in name
                or f"model.layers.2." in name
                or f"model.layers.3." in name
                or f"model.layers.4." in name
                or f"model.layers.5." in name
            ):
                continue
            if "embed" in name or "lm_head" in name:
                continue
            else:
                logger.info(f"freezing {name}")
                param.requires_grad = False

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < tokenizer.model_max_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["prefix"],
            data_point["plain_text"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        user_prompt = prompter.generate_prompt(data_point["prefix"])
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]
        return tokenized_full_prompt

    if data_args.data_path.endswith(".json") or data_args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_args.data_path)
    else:
        data = load_dataset(data_args.data_path)

    if data_args.val_data_path is not None and (
        data_args.val_data_path.endswith(".json")
        or data_args.val_data_path.endswith(".jsonl")
    ):
        valid_data = load_dataset("json", data_files=data_args.val_data_path)
    else:
        valid_data = None

    if data_args.test_data_path is not None and (
        data_args.test_data_path.endswith(".json")
        or data_args.test_data_path.endswith(".jsonl")
    ):
        _test_data = load_dataset("json", data_files=data_args.test_data_path)
    else:
        _test_data = None

    if training_args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=training_args.val_set_size, shuffle=True, seed=42
        )
        train_val_data = train_val.map(
            generate_and_tokenize_prompt,
            batched=False,
            num_proc=training_args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"generate_and_tokenize_prompt",
        )
        train_data = train_val_data["train"]
        val_data = train_val_data["test"]

    elif valid_data is not None:
        train_data = data["train"].map(
            generate_and_tokenize_prompt,
            batched=False,
            num_proc=training_args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"generate_and_tokenize_prompt",
        )
        val_data = valid_data["train"].map(
            generate_and_tokenize_prompt,
            batched=False,
            num_proc=training_args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"generate_and_tokenize_prompt_valid",
        )
    else:
        train_data = data["train"].map(
            generate_and_tokenize_prompt,
            batched=False,
            num_proc=training_args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"generate_and_tokenize_prompt",
        )
        val_data = None

    if _test_data is not None:
        test_data = _test_data["train"].map(
            generate_and_tokenize_prompt,
            batched=False,
            num_proc=training_args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"generate_and_tokenize_prompt_test",
        )
    else:
        test_data = None

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    def postprocess_text(
        preds: List[str], labels: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Post-process text by removing special tokens and extra whitespace."""

        # > Remove '!' and strip whitespace
        preds = [pred.replace("!", "").strip() for pred in preds]
        labels = [label.replace("!", "").strip() for label in labels]

        # > Remove multiple spaces
        preds = [" ".join(pred.split()) for pred in preds]
        labels = [" ".join(label.split()) for label in labels]

        return preds, labels

    def extract_parts(text: str, task: str = None) -> Tuple[str, str, str, str]:
        """Extract different parts from the structured text.
        input text does not include prefix(-100 parts)
        """

        if task == "asr+ser":
            try:
                emo = text.split(">")[0].split("<")[1].strip()
                transcription = text.split(">")[1].strip()
            except IndexError:
                emo = text.split(">")[0].strip()
                transcription = text.split(">")[1].strip()

            return emo, transcription

        try:
            cur_emo = text.split(">")[0].split("<")[-1].strip()

            cur_text = (
                text.split(">")[1].split("EmoSDS:")[0].strip()
            )

            after_answer = text.split("EmoSDS:")[1].strip()
            response_emo = after_answer[
                after_answer.find("<") + 1 : after_answer.find(">")
            ].strip()
        except:
            raise Exception(f"text: {text}")

        response_text = after_answer[after_answer.find(">") + 1 :].strip()
        if response_text == "":
            raise Exception(f"[{text}]")

        return (
            cur_emo,
            cur_text,
            response_emo,
            response_text,
        )

    def compute_metrics(eval_preds) -> Dict:
        import evaluate
        import numpy as np
        from collections import defaultdict

        bleu_metric = evaluate.load("sacrebleu")
        bertscore = evaluate.load("bertscore")
        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")
        f1_metric = evaluate.load("f1")

        preds, labels = eval_preds
        preds = np.array(preds)
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.argmax(preds, axis=-1)
        preds = preds.reshape(-1, preds.shape[-1])

        # > Create masked predictions array
        masked_preds = []
        for pred_seq, label_seq in zip(preds, labels):
            if training_args.train_task in [
                "unified",
                "asr+ser",
                "asr",
            ]:
                label_seq = np.roll(
                    label_seq, -1
                )

            # > Get indices where labels are not -100
            valid_indices = label_seq != -100

            # > Keep only the tokens at valid positions
            masked_pred = pred_seq[valid_indices]
            masked_preds.append(masked_pred)

        # > Convert to proper format for batch_decode
        masked_preds = np.array(
            [
                np.pad(
                    seq,
                    (0, max(len(p) for p in masked_preds) - len(seq)),
                    "constant",
                    constant_values=tokenizer.pad_token_id,
                )
                for seq in masked_preds
            ]
        )
        decoded_preds = tokenizer.batch_decode(masked_preds, skip_special_tokens=True)

        # > replace -100 to pad token
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = {}
        if training_args.train_task == "asr":
            wer_score = wer_metric.compute(
                predictions=decoded_preds, references=decoded_labels
            )
            cer_score = cer_metric.compute(
                predictions=decoded_preds, references=decoded_labels
            )

            result = {
                "wer": wer_score * 100,
                "cer": cer_score * 100,
            }
        elif training_args.train_task == "asr+ser":
            # > Extract parts from predictions
            emo_preds, texts_preds = [], []

            for pred in decoded_preds:
                emo, text = extract_parts(pred, task="asr+ser")

                # > Convert to lowercase for matching
                emo = emo.lower()

                # > Map to indices and validate
                emo_idx = valid_emotions.get(emo, -1)

                emo_preds.append(emo_idx)
                texts_preds.append(text)

            # > Extract parts from labels
            emo_labels, texts_labels = [], []

            for i, label in enumerate(decoded_labels):
                emo, text = extract_parts(label, task="asr+ser")

                # > Convert to lowercase for matching
                emo = emo.lower()

                # > Map to indices and validate
                emo_idx = valid_emotions.get(emo, -1)

                emo_labels.append(emo_idx)
                texts_labels.append(text)

            # > Calculate WER
            wer_text = wer_metric.compute(
                predictions=texts_preds, references=texts_labels
            )

            cer_text = cer_metric.compute(
                predictions=texts_preds, references=texts_labels
            )

            f1 = f1_metric.compute(
                predictions=emo_preds,
                references=emo_labels,
                average="weighted",
            )

            result = {
                "wer_text": wer_text * 100,
                "cer_text": cer_text * 100,
                "emo_f1": f1["f1"] * 100,
            }
        elif training_args.train_task == "unified":
            # > Extract parts from predictions
            (
                cur_emo_preds,
                cur_texts_preds,
                response_emo_preds,
                response_texts_preds,
            ) = (
                [],
                [],
                [],
                [],
            )

            for pred in decoded_preds:
                cur_emo, cur_text, response_emo, response_text = extract_parts(pred)

                # > Convert to lowercase for matching
                cur_emo = cur_emo.lower()
                response_emo = response_emo.lower()

                # > Map to indices and validate
                cur_emo_idx = valid_emotions.get(cur_emo, -1)
                response_emo_idx = valid_emotions.get(response_emo, -1)

                if cur_emo_idx != -1:
                    valid_cur_emo_count += 1
                if response_emo_idx != -1:
                    valid_response_emo_count += 1
                total_count += 1

                cur_emo_preds.append(cur_emo_idx)
                cur_texts_preds.append(cur_text)
                response_emo_preds.append(response_emo_idx)
                response_texts_preds.append(response_text)

            # > Extract parts from labels
            (
                cur_emo_labels,
                cur_texts_labels,
                response_emo_labels,
                response_texts_labels,
            ) = (
                [],
                [],
                [],
                [],
            )

            for i, label in enumerate(decoded_labels):
                cur_emo, cur_text, response_emo, response_text = extract_parts(
                    label
                )

                # > Convert to lowercase for matching
                cur_emo = cur_emo.lower()
                response_emo = response_emo.lower()

                # > Map to indices and validate
                cur_emo_idx = valid_emotions.get(cur_emo, -1)
                response_emo_idx = valid_emotions.get(response_emo, -1)

                cur_emo_labels.append(cur_emo_idx)
                cur_texts_labels.append(cur_text)
                response_emo_labels.append(response_emo_idx)
                response_texts_labels.append(response_text)

            f1_results_cur_emo = f1_metric.compute(
                predictions=cur_emo_preds,
                references=cur_emo_labels,
                average="weighted",
            )
            f1_results_response_emo = f1_metric.compute(
                predictions=response_emo_preds,
                references=response_emo_labels,
                average="weighted",
            )
            bleu_results_response_text = bleu_metric.compute(
                predictions=response_texts_preds,
                references=[[label] for label in response_texts_labels],
            )
            wer_cur_text = wer_metric.compute(
                predictions=cur_texts_preds, references=cur_texts_labels
            )
            cer_cur_text = cer_metric.compute(
                predictions=cur_texts_preds, references=cur_texts_labels
            )
            bertscore_results = bertscore.compute(
                predictions=response_texts_preds,
                references=[[label] for label in response_texts_labels],
                lang="en",
            )
            rouge_results = rouge.compute(
                predictions=response_texts_preds, references=response_texts_labels
            )
            meteor_results = meteor.compute(
                predictions=response_texts_preds, references=response_texts_labels
            )

            result = {
                "weighted_f1_cur_emo": f1_results_cur_emo["f1"] * 100,
                "weighted_f1_res_emo": f1_results_response_emo["f1"] * 100,
                "bleu_res_text": bleu_results_response_text["score"],
                "wer_cur_text": wer_cur_text * 100,
                "cer_cur_text": cer_cur_text * 100,
                "bertscore_f1": (
                    sum(bertscore_results["f1"]) / len(bertscore_results["f1"])
                )
                * 100,
                "rouge_L": rouge_results["rougeL"],
                "meteor": meteor_results["meteor"],
            }

        return result

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=val_data if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.initial_global_step != 0:
        logger.info(f"Set initial global step={training_args.initial_global_step}")
        trainer.state.global_step = training_args.initial_global_step

    # > Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        print("finished training")
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_data)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_data))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )

    # > Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(val_data)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(val_data))

        import math

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Test
    if training_args.do_predict and test_data is not None:
        logger.info("*** Test ***")

        metrics = trainer.predict(
            test_dataset=test_data,
        ).metrics

        import math

        try:
            perplexity = math.exp(metrics["test_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    from pathlib import Path
    import sys

    # > Add EmoSDS/ to path
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    sys.path.append(str(project_root))

    train()
