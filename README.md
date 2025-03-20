# EmoSDS
This is a repository of paper: EmoSDS: Emotionally Adaptive Spoken dialogue System

<img width="600" alt="image" src="asset/emosds_only_model_2.png">

## Abstract
In recent years, advancements in artificial intelligence, speech, and natural lan-
guage processing technology have enhanced spoken dialogue systems (SDS), enabling
natural voice-based human-computer interaction. However, discrete token-based LLMs
in emotionally adaptive SDS focus on lexical content while overlooking essential paralin-
guistic cues for emotion expression. Existing methods use external emotion predictors to
compensate but introduce computational overhead and fail to fully integrate paralinguis-
tic features with linguistic context. Moreover, the lack of high-quality emotional speech
datasets limits models’ ability to learn expressive emotional cues. To address these chal-
lenges, we propose EmoSDS, a unified SDS framework that integrates speech and emotion
recognition by leveraging self-supervised learning (SSL) features. Our three-stage train-
ing pipeline enables the LLM to learn both discrete linguistic content and continuous
paralinguistic features, improving emotional expressiveness and response naturalness.
Additionally, we construct EmoSC, a dataset combining GPT-generated dialogues with
emotional voice conversion data, ensuring greater emotional diversity and a balanced
sample distribution across emotion categories. Experimental results show that EmoSDS
outperforms existing models in emotional alignment and response generation achieving a
minimum 2.86% increase in text generation metrics, enhancing the LLM’s ability to interpret
emotional and textual cues for more expressive and contextually appropriate responses.

## How to train

Our google drive: https://drive.google.com/drive/folders/147S91ceHFPA0m3CEOb1jmc82dnpLlzaq?usp=sharing

Prerequisites:

1. python 3.8 env (refer to requirements.txt for all dependencies)

2. LlaMA 3.2 3B

	you can download it from: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

3. Pretrained K-means model

	Download it (LibriSpeech100-360-500_wavlm_k1000_L6.pt) under utils/speech2unit/, from our google drive.


※ You can also access stage 1,2 model checkpoints from our google drive.

### Stage 1
Download LibriSpeech train-clean-100 dataset.

```python
# create stage 1 data
python3 utils/build_data.py asr --librispeech-dir /path/to/your/librispeech/data

# train
bash scripts/asr_sft.sh
```

### Stage 2

Download ESD from [ESD official repository](https://github.com/HLTSingapore/Emotional-Speech-Data).

Download residual files (ESD_residual.zip) from our google drive, under the same parent directory of ESD.

```python
# create stage 2 data
python3 utils/build_data.py asr+ser --esd-dir /path/to/your/ESD/data --residual

# train
bash scripts/asr_ser_sft.sh # in the script, you should specify stage 1 checkpoint path via METAROOT
```

### Stage 3

Download synthesized dialogue data (EmoSC_dialogues.jsonl) from our google drive.

```python
# create stage 3 data (EmoSC)
python3 utils/build_data.py unified --esd-dir /path/to/your/ESD/data --esd-syn-path /path/to/dialogue/data

# train
bash scripts/unified_sft.sh # in the script, you should specify stage 2 checkpoint path via METAROOT
```

## Acknowledgement
Portions of the research in this paper used the ESD Database made available by the HLT lab, National University of Singapore, Singapore.

The codes are derived and modified from SpeechGPT
