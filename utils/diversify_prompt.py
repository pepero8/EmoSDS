import json
import typer
from typing_extensions import Annotated

ASR_SER = " Valid emotions are: <anger>, <happiness>, <neutral>, <sadness>, <surprise>."


def diversify_prompt(file, out, prompts_path, asr_ser):

    prompts = []
    with open(prompts_path, "r") as prompts_path:
        for line in prompts_path:
            prompts.append(line)

    with open(file, "r") as f:
        with open(out, "w") as out:
            prompt_idx = 0
            for line in f:
                data = json.loads(line)
                
                if asr_ser:
                    prefix_input = data["prefix"].split("prompt:")[1]
                    prefix_input = ASR_SER + prefix_input
                else:
                    prefix_input = data["prefix"].split(
                        "Transcribe following speech input: "
                    )[1]
                new_prefix = prompts[prompt_idx] + prefix_input
                prompt_idx = (prompt_idx + 1) % len(prompts)
                data["prefix"] = new_prefix

                json.dump(data, out, ensure_ascii=False)
                out.write("\n")


def main(
    file: Annotated[
        str,
        typer.Argument(help="path to dataset"),
    ],
    out: Annotated[
        str,
        typer.Argument(help="output path"),
    ],
    prompts_path: Annotated[
        str,
        typer.Option(help="path to asr prompts txt file"),
    ] = None,
    asr_ser: Annotated[
        bool,
        typer.Option(help="whether it's asr+ser dataset"),
    ] = False,
):
    diversify_prompt(file, out, prompts_path, asr_ser)


if __name__ == "__main__":
    typer.run(main)
