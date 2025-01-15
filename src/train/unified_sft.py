# stage3: dialogue finetuning
import math
import sys
sys.path.append("/home/jhwan98/EmoSDS/SpeechGPT/speechgpt")
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from datasets import load_dataset
from datasets import config
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint
# from speechgpt.utils.prompter import Prompter
from utils.prompter import Prompter
import os
import logging
# from datasets import load_metric
import evaluate
from typing import Dict, List, Tuple
import numpy as np

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


@dataclass
class DataArguments:
    data_path: str = field(
        default="", 
        metadata={"help": "Path to the training data."})
    prompt_template_name: str = field(
        default="alpaca",
        metadata={"help": "prompt_template_name"},
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

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the tokenized data"},
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
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
        default=0,
        metadata={"help": "initial_global_step"}
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses() # training_args에 bf_16같은 arguments도 들어감

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set different cache directory
    config.HF_DATASETS_CACHE = training_args.cache_dir # added by jaehwan

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    prompter = Prompter()

    # model = LlamaForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    ).to(
        torch.device(training_args.device)
    )  # added by jaehwan

    # tokenizer = LlamaTokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     model_max_length=training_args.model_max_length,
    #     padding_side="right",
    #     use_fast=False,
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )  # added by jaehwan

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    # Extend vocab for speech units
    if '<sosp>' not in tokenizer.get_vocab():
        units_size=1000
        logger.info(f"Add special unit tokens <0>-<{units_size-1} to tokenizer.vocab")
        new_tokens = [f"<{x}>" for x in range(units_size)] + ['<sosp>','<eosp>']
        tokenizer.add_tokens(new_tokens)
        logger.info(f"Add style tokens to tokenizer.vocab")
        # style_tokens = ['<neutral>', '<angry>', '<cheerful>', '<sad>', '<excited>', '<friendly>', '<terrified>', '<shouting>', '<unfriendly>', '<whispering>', '<hopeful>'] # for StyleTalk
        style_tokens = ['<anger>', '<disgust>', '<fear>', '<happiness>', '<neutral>', '<sadness>', '<surprise>'] # for DailyTalk
        tokenizer.add_tokens(style_tokens)
    # for token in ['<sosp>','<eosp>']:
    # for token in ['<sosp>','<eosp>', '<neutral>', '<angry>', '<cheerful>', '<sad>', '<excited>', '<friendly>', '<terrified>', '<shouting>', '<unfriendly>', '<whispering>', '<hopeful>']: # for StyleTalk
    for token in ['<anger>', '<disgust>', '<fear>', '<happiness>', '<neutral>', '<sadness>', '<surprise>']: # for DailyTalk
        if token not in tokenizer.get_vocab():
            logger.info(f"Add special unit tokens {token} to tokenizer.vocab")
            tokenizer.add_tokens([token])

    # resize embedding
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
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
        '''
        moss-style instructions
        '''
        full_prompt = prompter.generate_prompt(
            data_point["prefix"],
            data_point["plain_text"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        user_prompt = prompter.generate_prompt(
            data_point["prefix"]
        )
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
        return tokenized_full_prompt

    if data_args.data_path.endswith(".json") or data_args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_args.data_path)
    else:
        data = load_dataset(data_args.data_path)

    # tokenized_cache_file_names = {
    #     "train":os.path.join(training_args.cache_dir, 'tokenized', 'train', 'processed_train.arrow'),
    #     "test":os.path.join(training_args.cache_dir, 'tokenized', 'valid', 'processed_valid.arrow'),
    #     }

    if training_args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=training_args.val_set_size, shuffle=True, seed=42
        )
        train_val_data = (
            train_val.map(
                generate_and_tokenize_prompt,
                batched=False,
                num_proc=training_args.preprocessing_num_workers,
                load_from_cache_file=True,
                # cache_file_names=tokenized_cache_file_names, # removed by jaehwan
                desc=f"generate_and_tokenize_prompt",
                )
        )
        train_data = train_val_data["train"]
        val_data = train_val_data["test"]

    else:
        train_data = data["train"].map(
            generate_and_tokenize_prompt,
            batched=False,
            num_proc=training_args.preprocessing_num_workers,
            load_from_cache_file=True,
            # cache_file_names=tokenized_cache_file_names, # removed by jaehwan
            desc=f"generate_and_tokenize_prompt",
        )
        val_data = None

    data_collator = DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )

    # def postprocess_text(
    #     preds: List[str], labels: List[str]
    # ) -> Tuple[List[str], List[str]]:
    #     """Post-process text by removing special tokens and extra whitespace."""
    #     preds = [pred.strip() for pred in preds]
    #     labels = [label.strip() for label in labels]

    #     return preds, labels

    def postprocess_text(
        preds: List[str], labels: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Post-process text by removing special tokens and extra whitespace."""
        # preds = [pred.strip() for pred in preds]
        # labels = [label.strip() for label in labels]

        # Remove '!' and strip whitespace
        preds = [pred.replace("!", "").strip() for pred in preds]
        labels = [label.replace("!", "").strip() for label in labels]

        # Remove multiple spaces
        preds = [" ".join(pred.split()) for pred in preds]
        labels = [" ".join(label.split()) for label in labels]

        return preds, labels

    def extract_parts(text: str) -> Tuple[str, str, str, str]:
        """Extract different parts from the structured text.
           input text does not include prefix(-100 parts)
        """
        try:
            # Extract cur_style (part before first ']')
            cur_style = text.split(">")[0].split("<")[1].strip()
            # cur_style = text.split("]")[0].strip()

            # Extract cur_text (part between first ']' and 'Answer:')
            cur_text = text.split(">")[1].split("Answer:")[0].strip()
            # cur_text = text.split("]")[1].split("Answer:")[0].strip()

            # Extract response_style (part between '[' and ']' after 'Answer:')
            after_answer = text.split("Answer:")[1].strip()
            response_style = after_answer[
                # after_answer.find("[") + 1 : after_answer.find("]")
                after_answer.find("<") + 1 : after_answer.find(">")
            ].strip()
        except:
            raise Exception(f"text: {text}")

        # Extract response_text (remaining part after response_style)
        response_text = after_answer[after_answer.find(">") + 1 :].strip()
        # response_text = after_answer[after_answer.find("]") + 1 :].strip()
        if response_text == "":
            raise Exception(f"[{text}]")

        return cur_style, cur_text, response_style, response_text
        # except Exception:
        # Return empty strings if parsing fails
        # return "", "", "", ""

    def compute_metrics(eval_preds) -> Dict:
        """
        Compute BLEU, BERT score along with other metrics.
        Args:
            eval_preds: tuple containing predictions and labels from Trainer
        Returns:
            dict containing metrics
        """

        # >> for StyleTalk
        # VALID_STYLES = {
        #     'neutral': 0,
        #     'angry': 1,
        #     'cheerful': 2,
        #     'sad': 3,
        #     'excited': 4, 
        #     'friendly': 5,
        #     'terrified': 6,
        #     'shouting': 7,
        #     'unfriendly': 8, 
        #     'whispering': 9,
        #     'hopeful': 10
        # }

        # >> for DailyTalk
        VALID_STYLES = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'happiness': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }

        bleu_metric = evaluate.load("sacrebleu")
        bertscore = evaluate.load("bertscore")
        wer_metric = evaluate.load("wer")
        acc_metric = evaluate.load("accuracy")

        preds, labels = eval_preds
        # print(f"^^7 fuck yeah {labels[21]}")
        preds = np.array(preds)
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode generated tokens to text
        # print("preds:", len(preds), len(preds[0]), len(preds[1]))
        # print("preds shape:", preds.shape)
        preds = np.argmax(preds, axis=-1)
        preds = preds.reshape(-1, preds.shape[-1])

        # Create masked predictions array
        masked_preds = []
        for pred_seq, label_seq in zip(preds, labels):
            temp_label = np.roll(label_seq, -1) # 이거 안해주면 style 토큰이 제대로 추출되지 않음
            # Get indices where labels are not -100
            valid_indices = temp_label != -100
            # Keep only the tokens at valid positions
            masked_pred = pred_seq[valid_indices]
            masked_preds.append(masked_pred)

        # Convert to proper format for batch_decode
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

        # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id) # -100 부분은 pad token으로 교체
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Post-process predictions and labels
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        # print(f"^^7 preds: [{decoded_preds}], labels: [{decoded_labels}]")

        # Extract parts from predictions
        cur_styles_preds, cur_texts_preds, response_styles_preds, response_texts_preds = (
            [],
            [],
            [],
            [],
        )

        # Style validation counters
        valid_cur_style_count = 0
        valid_response_style_count = 0
        total_count = 0

        for pred in decoded_preds:
            cur_style, cur_text, response_style, response_text = extract_parts(pred)

            # Convert to lowercase for matching
            cur_style = cur_style.lower()
            response_style = response_style.lower()

            # Map to indices and validate
            cur_style_idx = VALID_STYLES.get(cur_style, -1)
            response_style_idx = VALID_STYLES.get(response_style, -1)

            if cur_style_idx != -1:
                valid_cur_style_count += 1
            if response_style_idx != -1:
                valid_response_style_count += 1
            total_count += 1

            # Validate styles
            # if cur_style.lower() in VALID_STYLES:
            #     valid_cur_style_count += 1
            # if response_style.lower() in VALID_STYLES:
            #     valid_response_style_count += 1
            # total_count += 1

            cur_styles_preds.append(cur_style_idx)
            cur_texts_preds.append(cur_text)
            response_styles_preds.append(response_style_idx)
            response_texts_preds.append(response_text)

        # Extract parts from labels
        (
            cur_styles_labels,
            cur_texts_labels,
            response_styles_labels,
            response_texts_labels,
        ) = (
            [],
            [],
            [],
            [],
        )

        # for label in decoded_labels:
        for i, label in enumerate(decoded_labels):
            cur_style, cur_text, response_style, response_text = extract_parts(label)

            # Convert to lowercase for matching
            cur_style = cur_style.lower()
            response_style = response_style.lower()

            # Map to indices
            cur_style_idx = VALID_STYLES.get(cur_style, -1)
            response_style_idx = VALID_STYLES.get(response_style, -1)

            cur_styles_labels.append(cur_style_idx)
            cur_texts_labels.append(cur_text)
            response_styles_labels.append(response_style_idx)
            response_texts_labels.append(response_text)

        # Calculate style validation percentages
        cur_style_valid_percentage = (valid_cur_style_count / total_count) * 100
        response_style_valid_percentage = (valid_response_style_count / total_count) * 100

        print(f"^^7 pred: [{decoded_preds[21]}]")
        print(f"^^7\n   [{cur_styles_preds[21]}] -> [{cur_styles_labels[21]}]\n \
                        [{cur_texts_preds[21]}] -> [{cur_texts_labels[21]}]\n \
                        [{response_styles_preds[21]}] -> [{response_styles_labels[21]}]\n \
                        [{response_texts_preds[21]}] -> [{response_texts_labels[21]}]\n")
        # raise Exception("fuck yeah")

        # Compute BLEU score for current style
        acc_results_cur_style = acc_metric.compute(
            predictions=cur_styles_preds,
            references=cur_styles_labels
            # references=[[label] for label in cur_styles_labels]
        )
        # bleu_results_cur_style = bleu_metric.compute(
        #     predictions=cur_styles_preds, references=[[label] for label in cur_styles_labels]
        # )

        # Compute BLEU score for reponse style
        acc_results_response_style = acc_metric.compute(
            predictions=response_styles_preds,
            references=response_styles_labels
            # references=[[label] for label in response_styles_labels],
            # references=[response_styles_labels[21]],
        )
        # bleu_results_response_style = bleu_metric.compute(
        #     predictions=response_styles_preds,
        #     references=[[label] for label in response_styles_labels],
        #     # references=[response_styles_labels[21]],
        # )

        # Compute BLEU score for current text
        bleu_results_cur_text = bleu_metric.compute(
            predictions=cur_texts_preds,
            references=[[label] for label in cur_texts_labels],
        )

        # Compute BLEU score for response text
        bleu_results_response_text = bleu_metric.compute(
            predictions=response_texts_preds,
            references=[[label] for label in response_texts_labels],
        )

        # Compute WER score for current text
        wer_cur_text = wer_metric.compute(
            predictions=cur_texts_preds, references=cur_texts_labels
        )

        # Compute WER score for response text
        wer_response_text = wer_metric.compute(
            predictions=response_texts_preds, references=response_texts_labels
        )

        # Compute BERTscore for response text
        bertscore_results = bertscore.compute(
            predictions=response_texts_preds,
            references=[[label] for label in response_texts_labels],
            lang="en",
        )

        # Return all metrics
        result = {
            "acc_cur_style": acc_results_cur_style["accuracy"] * 100,
            "acc_res_style": acc_results_response_style["accuracy"] * 100,
            "bleu_cur_text": bleu_results_cur_text["score"],
            "bleu_res_text": bleu_results_response_text["score"],
            "wer_cur_text": wer_cur_text * 100,
            "wer_res_text": wer_response_text * 100,
            "bertscore_f1": (
                sum(bertscore_results["f1"]) / len(bertscore_results["f1"])
            ) * 100,
            "valid_cur_style_percentage": cur_style_valid_percentage,
            "valid_response_style_percentage": response_style_valid_percentage
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

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_data)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_data))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_data)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(val_data) # added by jaehwan
        # metrics["eval_samples"] = min(max_eval_samples, len(eval_data))
        metrics["eval_samples"] = min(
            max_eval_samples, len(val_data)
        )  # added by jaehwan
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    train()
