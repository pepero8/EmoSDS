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
    # >> Load WavLM
    processor = AutoProcessor.from_pretrained(wavlm_model_name)
    wavlm = WavLMModel.from_pretrained(wavlm_model_name)

    # >> Load KMeans model
    kmeans = ApplyKmeans(kmeans_path)

    return processor, wavlm, kmeans


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        print(f"size of C of kmeans model: {self.C.shape}")  # just for debugging
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
        km_path = os.path.join(ckpt_dir, "LibriSpeech_wavlm_k1000_L12.pt")

        processor, wavlm, kmeans = load_models(
            wavlm_model_name=encoder_name, kmeans_path=km_path
        )

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

    def extract_wavlm_features(
        self,
        audio_path,
        layer_num=12,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.wavlm = self.wavlm.to(device)
        self.wavlm.eval()

        waveform, sample_rate = torchaudio.load(audio_path)

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
            ]  # (batch_size, sequence_length, 1024)
            features = layer_output.flatten(
                end_dim=-2
            )  # (sequence_length, hidden_size)

        return features.cpu().numpy()

    def __call__(self, path, merged=True):

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


if __name__ == "__main__":
    raise RuntimeError("Direct execution is not currently supported (˘･_･˘)")
