import logging
import os
import sys
import joblib

import torch
import numpy as np
import torchaudio
from transformers import WavLMModel, AutoProcessor

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("generate_pseudo_language")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(
    wavlm_model_name="patrickvonplaten/wavlm-libri-clean-100h-large",
    kmeans_path="LibriSpeech_wavlm_k1000_L12.pt",
):
    processor = AutoProcessor.from_pretrained(wavlm_model_name)
    wavlm = WavLMModel.from_pretrained(wavlm_model_name)

    kmeans = ApplyKmeans(kmeans_path)

    return processor, wavlm, kmeans


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = (
            self.km_model.cluster_centers_.transpose()
        )
        self.Cnorm_np = (self.C_np**2).sum(
            0, keepdims=True
        )

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            self.C = self.C.to(x)
            self.Cnorm = self.Cnorm.to(x)
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


class Speech2UnitCustom(torch.nn.Module):
    def __init__(self, ckpt_dir):
        super().__init__()

        encoder_name = "patrickvonplaten/wavlm-libri-clean-100h-large"
        km_path = os.path.join(
            ckpt_dir,
            "LibriSpeech100-360-500_wavlm_k1000_L6.pt",
        )

        processor, wavlm, kmeans = load_models(
            wavlm_model_name=encoder_name, kmeans_path=km_path
        )

        self.layer_num = 6
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

    @staticmethod
    def downsample(cluster_ids):
        downsampled_clusters = cluster_ids[::4]
        duration_list = [4] * len(downsampled_clusters)
        # Handle the last segment if total length is not divisible by 4
        if len(cluster_ids) % 4 != 0:
            duration_list[-1] = len(cluster_ids) % 4
        return downsampled_clusters, duration_list

    def extract_wavlm_features(
        self,
        audio_path,
        layer_num=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.wavlm = self.wavlm.to(device)
        self.wavlm.eval()

        if layer_num is None:
            layer_num = self.layer_num

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except RuntimeError as e:
            raise e

        # >> Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # >> Resample to 16kHz if needed
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=target_sample_rate
            )

        # >> Convert to numpy array and flatten
        audio_input = waveform.squeeze().numpy()

        inputs = self.processor(
            audio_input,
            sampling_rate=target_sample_rate,
            return_tensors="pt",
            padding=True,
        )

        # >> Move inputs to device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # >> Extract features
        with torch.no_grad():
            outputs = self.wavlm(**inputs, output_hidden_states=True)
            layer_output = outputs.hidden_states[
                layer_num
            ]
            features = layer_output.flatten(
                end_dim=-2
            )

        return features.cpu().numpy()

    def get_cluster_embedding(self, cluster_id):
        # Get the cluster center (embedding) for a given cluster ID
        return self.kmeans.C_np[:, cluster_id]

    def __call__(self, path, merged=True, downsample=False, residual=False):
        if residual:
            residual_path = str(path).replace(
                "EmotionSpeechDataset", "EmotionSpeechDataset_residual_6L-1000k"
            )
            residual_path = os.path.splitext(residual_path)[0] + ".npy"
            residual_np = np.load(residual_path)
            residual_length = residual_np.shape[0]

        try:
            features = self.extract_wavlm_features(path)
        except RuntimeError as e:
            raise e

        cluster_ids = self.kmeans(features).tolist()

        if merged:
            new_cluster_list, duration_list = self.merge_duplicates(cluster_ids)
        elif downsample:
            new_cluster_list, duration_list = self.downsample(cluster_ids)
        else:
            new_cluster_list = cluster_ids

        units = "<sosp>" + "".join([f"<{str(x)}>" for x in new_cluster_list]) + "<eosp>"

        if residual:
            return units, residual_length, residual_path
        else:
            return units
