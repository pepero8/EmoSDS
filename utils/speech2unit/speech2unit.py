from typing import List, Union
import logging
import os
import sys
import joblib
import fire
# import fairseq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from einops import rearrange
import re
import numpy as np
from functools import partial
import torch.multiprocessing as mp
import torchaudio
import glob
import tqdm
import argparse
from torchaudio.functional import resample
from transformers import WavLMModel, AutoProcessor

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger('generate_pseudo_language')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(
    wavlm_model_name="patrickvonplaten/wavlm-libri-clean-100h-large",
    kmeans_path="LibriSpeech_wavlm_k1000_L12.pt",
):
    """
    Load both WavLM and KMeans models

    Args:
        wavlm_model_name (str): Name of the WavLM model to load
        kmeans_path (str): Path to the pretrained KMeans model

    Returns:
        tuple: (WavLM processor, WavLM model, KMeans model)
    """
    # Load WavLM
    processor = AutoProcessor.from_pretrained(wavlm_model_name)
    wavlm = WavLMModel.from_pretrained(wavlm_model_name)

    # Load KMeans model
    kmeans = ApplyKmeans(kmeans_path)
    # kmeans = joblib.load(kmeans_path)

    return processor, wavlm, kmeans


# class FeatureReader(object):
#     def __init__(self, ckpt_path, layer, max_chunk=1600000, fp16=False, sampling_rate=16000):
#         (
#             model,
#             cfg,
#             task,
#         ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
#         self.model = model[0].eval().to(DEVICE)
#         self.task = task
#         self.layer = layer
#         self.max_chunk = max_chunk
#         self.fp16 = fp16
#         if fp16:
#             self.model.half()

#         self.layer_shift = 0
#         self.target_sample_hz = sampling_rate

#         logger.info(f"TASK CONFIG:\n{self.task.cfg}")

#     def read_audio(self, path):
#         wav, sr = torchaudio.load(path)
#         if sr != self.target_sample_hz:
#             wav = resample(wav, sr, self.target_sample_hz)
#         return wav

#     @torch.no_grad()
#     def get_feats(self, waveform):
#         x = waveform
#         with torch.no_grad():
#             if self.fp16:
#                 x = x.half().cuda()
#             else:
#                 x = x.float().cuda()
#             if self.task.cfg.normalize:
#                 x = F.layer_norm(x, x.shape)
#             x = x.view(1, -1)

#             feat = []
#             for start in range(0, x.size(1), self.max_chunk):
#                 x_chunk = x[:, start: start + self.max_chunk]
#                 feat_chunk, _ = self.model.extract_features(
#                         source=x_chunk,
#                         padding_mask=None,
#                         mask=False,
#                         output_layer=self.layer + self.layer_shift,
#                 )

#                 feat.append(feat_chunk)
#         if len(feat) == 0:
#             return torch.zeros(0, 0)
#         return torch.cat(feat, 1).squeeze(0)


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        print(f"size of C of kmeans model: {self.C.shape}") # added by jaehwan
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        # x: (num, ssl_hidden_dim)
        if isinstance(x, torch.Tensor):
            self.C = self.C.to(x)
            self.Cnorm = self.Cnorm.to(x)
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


# class Speech2Unit(torch.nn.Module):
#     def __init__(
#         self,
#         ckpt_dir,
#         layer=11,
#         max_chunk=1600000,
#         fp16=False,
#         sampling_rate=16000,
#         ):

#         """
#         Args:
#             ckpt_dir(str): path to hubert model dir(e.g. hubert_base_ls960.pt)
#             layer(int): feat from which layer of hubert models defauly by 9
#             max_chunk(int): default by 1600000
#             fp16(bool): default by False
#             sampling_rate(int): sampling_rate default by 16000
#         """
#         super().__init__()

#         ckpt_path = os.path.join(ckpt_dir, "mhubert_base_vp_en_es_fr_it3.pt")
#         km_path = os.path.join(ckpt_dir, "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")

#         self.feature_reader = FeatureReader(ckpt_path, layer, max_chunk, fp16, sampling_rate)
#         self.apply_kmeans = ApplyKmeans(km_path)

#     @staticmethod
#     def merge_duplicates(cluster_ids):
#         dup_cluster_list = []
#         duration_list = []
#         count = 1
#         for i in range(0, len(cluster_ids)):
#             if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i+1]:
#                 count += 1
#             else:
#                 dup_cluster_list.append(cluster_ids[i])
#                 duration_list.append(count)
#                 count = 1
#         return dup_cluster_list, duration_list

#     def __call__(self, path, merged=True):
#         waveform = self.feature_reader.read_audio(path).to(DEVICE)

#         feat = self.feature_reader.get_feats(waveform)
#         cluster_ids = self.apply_kmeans(feat).tolist()
#         dup_cluster_list, duration_list = self.merge_duplicates(cluster_ids)

#         merged_units = "<sosp>" + "".join([f"<{str(x)}>" for x in dup_cluster_list]) + "<eosp>"
#         unmerged_units = "<sosp>" + "".join([f"<{str(x)}>" for x in cluster_ids]) + "<eosp>"

#         if merged:
#             return merged_units
#         else:
#             return unmerged_units
#         # return {"continuous":feat, "units":dup_cluster_list, "duration":duration_list, "unmerged_units":cluster_ids}


class Speech2UnitCustom(torch.nn.Module):
    def __init__(
        self,
        ckpt_dir,
        layer=11,
        max_chunk=1600000,
        fp16=False,
        sampling_rate=16000,
    ):
        """
        Args:
            ckpt_dir(str): path to hubert model dir(e.g. hubert_base_ls960.pt)
            layer(int): feat from which layer of hubert models defauly by 9
            max_chunk(int): default by 1600000
            fp16(bool): default by False
            sampling_rate(int): sampling_rate default by 16000
        """
        super().__init__()

        encoder_name = "patrickvonplaten/wavlm-libri-clean-100h-large"
        km_path = os.path.join(ckpt_dir, "LibriSpeech_wavlm_k1000_L12.pt")

        processor, wavlm, kmeans = load_models(wavlm_model_name=encoder_name, kmeans_path=km_path)

        self.processor = processor
        self.wavlm = wavlm
        self.kmeans = kmeans

    @staticmethod
    def merge_duplicates(cluster_ids):
        dup_cluster_list = []
        duration_list = []
        count = 1
        for i in range(0, len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i + 1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                duration_list.append(count)
                count = 1
        return dup_cluster_list, duration_list

    def extract_wavlm_features(self, audio_path, layer_num=12, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Extract features from specific WavLM layer
        
        Args:
            audio_path (str): Path to audio file
            processor: WavLM processor
            wavlm: WavLM model
            layer_num (int): Which layer to extract features from (default: 12)
            device (str): Device to run the model on
            
        Returns:
            numpy.ndarray: Features from specified layer
        """
        # Move model to device
        self.wavlm = self.wavlm.to(device)
        self.wavlm.eval()

        # Load audio using torchaudio (supports both WAV and FLAC)
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=target_sample_rate
            )

        # Convert to numpy array and flatten
        audio_input = waveform.squeeze().numpy()

        # Load and preprocess audio
        # audio_input, sampling_rate = sf.read(audio_path)
        # if len(audio_input.shape) > 1:
        #     audio_input = audio_input.mean(axis=1)

        inputs = self.processor(
            audio_input,
            # sampling_rate=sampling_rate,
            sampling_rate=target_sample_rate,
            return_tensors="pt",
            padding=True
        )

        # Move inputs to device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.wavlm(**inputs, output_hidden_states=True)
            # Get the specified layer's output
            layer_output = outputs.hidden_states[layer_num] # (batch_size, sequence_length, 1024)
            # Flatten the output for clustering
            features = layer_output.flatten(
                end_dim=-2
            )  # (sequence_length, hidden_size)

        return features.cpu().numpy()

    # def discretize_speech(self, audio_path, merged=True):
    #     """
    #     Discretize speech using WavLM embeddings and KMeans clustering

    #     Args:
    #         audio_path (str): Path to audio file
    #         processor: WavLM processor
    #         wavlm: WavLM model
    #         kmeans: Trained KMeans model

    #     Returns:
    #         numpy.ndarray: Discretized speech tokens (cluster assignments)
    #     """
    #     # Extract features from WavLM
    #     features = self.extract_wavlm_features(audio_path)
    #     cluster_ids = self.kmeans(features).tolist()
    #     dup_cluster_list, duration_list = self.merge_duplicates(cluster_ids)

    #     merged_units = (
    #         "<sosp>" + "".join([f"<{str(x)}>" for x in dup_cluster_list]) + "<eosp>"
    #     )
    #     unmerged_units = (
    #         "<sosp>" + "".join([f"<{str(x)}>" for x in cluster_ids]) + "<eosp>"
    #     )

    #     if merged:
    #         return merged_units
    #     else:
    #         return unmerged_units

    #     # Get cluster assignments
    #     # discrete_tokens = self.kmeans.predict(features)

    #     # return discrete_tokens

    def __call__(self, path, merged=True):
        """
        Discretize speech using WavLM embeddings and KMeans clustering

        Args:
            path (str): Path to audio file
            processor: WavLM processor
            wavlm: WavLM model
            kmeans: Trained KMeans model

        Returns:
            numpy.ndarray: Discretized speech tokens (cluster assignments)
        """
        # Extract features from WavLM
        features = self.extract_wavlm_features(path)
        cluster_ids = self.kmeans(features).tolist()
        dup_cluster_list, duration_list = self.merge_duplicates(cluster_ids)

        merged_units = (
            "<sosp>" + "".join([f"<{str(x)}>" for x in dup_cluster_list]) + "<eosp>"
        )
        unmerged_units = (
            "<sosp>" + "".join([f"<{str(x)}>" for x in cluster_ids]) + "<eosp>"
        )

        if merged:
            return merged_units
        else:
            return unmerged_units

        # Get cluster assignments
        # discrete_tokens = self.kmeans.predict(features)

        # return discrete_tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str)
    args = parser.parse_args()

    # ckpt_dir = "speechgpt/utils/speech2unit/"
    ckpt_dir = "./"

    # s2u = Speech2Unit(
    #     ckpt_dir=ckpt_dir
    # )

    s2u = Speech2UnitCustom(ckpt_dir=ckpt_dir)

    units = s2u(args.wav)
    print("[", units, "]")
