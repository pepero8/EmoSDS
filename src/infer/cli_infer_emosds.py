import typer
from typing_extensions import Annotated


NAME = "EmoSDS"

META_INSTRUCTION = """
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

Here's the prompt:\n\n"""

# USER_INSTRUCTION = "Identify speaking style of given speech: {units}. Provide only the style label > ["
DEFAULT_GEN_PARAMS = {
    "max_new_tokens": 4096,
    "min_new_tokens": 10,
    "temperature": 0.6,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.9,
}


def extract_text_between_tags(text, tag1="[EmoSDS] :", tag2="<eoa>"):
    import re

    pattern = f"{re.escape(tag1)}(.*?){re.escape(tag2)}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = ""
    return response


class EmoSDSInference:
    from typing import List

    def __init__(
        self,
        model_name_or_path: str,
        s2u_dir: str = "speechgpt/utils/speech2unit/",
        output_dir: str = "speechgpt/output/",
    ):
        from utils.speech2unit.speech2unit import Speech2UnitCustom
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.device = torch.device("cuda")

        self.meta_instruction = META_INSTRUCTION
        self.template = "{prompt}"

        self.s2u = Speech2UnitCustom(ckpt_dir=s2u_dir)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.half()
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        self.generate_kwargs = DEFAULT_GEN_PARAMS

        self.output_dir = output_dir

    def preprocess(
        self,
        raw_text: str,
    ):
        import os

        processed_parts = []
        for part in raw_text.split("<<Input>>:"):
            # for _ in raw_text:
            # if os.path.isfile(part.strip()) and os.path.splitext(part.strip())[-1] in [".wav", ".flac", ".mp4"]:
            if os.path.isfile(part.strip()) and os.path.splitext(part.strip())[-1] in [
                ".wav",
                ".flac",
                ".mp4",
            ]:
                processed_parts.append(
                    "Input: " + self.s2u(part.strip(), merged=False, downsample=True)
                )
            else:
                processed_parts.append(part)

        processed_text = "".join(processed_parts)

        prompt_seq = self.meta_instruction + self.template.format(prompt=processed_text)
        return prompt_seq

    def postprocess(
        self,
        response: str,
    ):

        question = extract_text_between_tags(response, tag1="[Human]", tag2="<eoh>")
        answer = extract_text_between_tags(
            response + "<eoa>", tag1=f"[SpeechGPT] :", tag2="<eoa>"
        )
        tq = (
            extract_text_between_tags(response, tag1="[SpeechGPT] :", tag2="; [ta]")
            if "[ta]" in response
            else ""
        )
        ta = (
            extract_text_between_tags(response, tag1="[ta]", tag2="; [ua]")
            if "[ta]" in response
            else ""
        )
        ua = (
            extract_text_between_tags(response + "<eoa>", tag1="[ua]", tag2="<eoa>")
            if "[ua]" in response
            else ""
        )

        return {
            "question": question,
            "answer": answer,
            "textQuestion": tq,
            "textAnswer": ta,
            "unitAnswer": ua,
        }

    def forward(self, prompts: List[str]):
        from transformers import GenerationConfig
        import torch

        with torch.no_grad():
            # preprocess
            preprocessed_prompts = []
            for prompt in prompts:
                preprocessed_prompts.append(self.preprocess(prompt))

            input_ids = self.tokenizer(
                preprocessed_prompts, return_tensors="pt", padding=True
            ).input_ids
            for input_id in input_ids:
                if input_id[-1] == 2:
                    input_id = input_id[:, :-1]

            input_ids = input_ids.to(self.device)

            generation_config = GenerationConfig(
                temperature=self.generate_kwargs["temperature"],
                top_p=self.generate_kwargs["top_p"],
                top_k=self.generate_kwargs["top_k"],
                do_sample=self.generate_kwargs["do_sample"],
                max_new_tokens=self.generate_kwargs["max_new_tokens"],
                min_new_tokens=self.generate_kwargs["min_new_tokens"],
            )

            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generated_ids = generated_ids.sequences
            responses = self.tokenizer.batch_decode(
                generated_ids.cpu(), skip_special_tokens=True
            )

            print("[", responses, "]")

        return

    def dump_wav(self, sample_id, pred_wav, prefix):
        import soundfile as sf

        sf.write(
            f"{self.output_dir}/wav/{prefix}_{sample_id}.wav",
            pred_wav.detach().cpu().numpy(),
            16000,
        )

    def __call__(self, input):
        return self.forward(input)

    def interact(self):
        import traceback

        prompt = str(input(f"Please talk with {NAME}:\n"))
        while prompt != "quit":
            try:
                self.forward([prompt])
            except Exception as e:
                traceback.print_exc()
                print(e)

            prompt = str(input(f"Please input prompts for {NAME}:\n"))


def main(
    model_path: Annotated[str, typer.Option(help="EmoSDS checkpoint path")],
    output_dir: Annotated[str, typer.Option(help="Path to save generated output")],
    interact: Annotated[
        bool, typer.Option(help="Whether to enable turn taking inference")
    ],
    input: Annotated[str, typer.Option(help="Path to inference input file")] = None,
    s2u_dir: Annotated[
        str, typer.Option(help="Speech2Unit path")
    ] = "utils/speech2unit",
):
    import os

    os.makedirs(output_dir, exist_ok=True)

    agent = EmoSDSInference(
        model_path,
        s2u_dir,
        output_dir,
    )

    if interact:
        agent.interact()
    else:
        if input is None:
            raise RuntimeError("Please provide input through --input option")
        print(f"WIP...")
        exit()


if __name__ == "__main__":
    from pathlib import Path
    import sys
    import logging

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    sys.path.append(str(project_root))

    logging.basicConfig()
    logging.root.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    typer.run(main)
