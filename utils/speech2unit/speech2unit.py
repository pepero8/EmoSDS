import logging
import os
from pathlib import Path
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
        self.km_model = joblib.load(km_path)  # > comment it for sim kmeans
        # self.km_model = torch.load(km_path, weights_only=True)  # > for sim kmeans
        # self.C = self.km_model.transpose(
        #     1, 0
        # )  # (ssl_feature_dim, num_clusters). cluster embeddings matrix # > for sim kmeans
        self.C_np = (
            self.km_model.cluster_centers_.transpose()
        )  # > comment it for sim kmeans
        # self.C_np = self.C.numpy()  # > for sim kmeans
        # self.Cnorm = (self.C**2).sum(0, keepdims=True)  # > for sim kmeans
        self.Cnorm_np = (self.C_np**2).sum(
            0, keepdims=True
        )  # > comment it for sim kmeans
        # self.Cnorm_np = self.Cnorm.numpy()  # > for sim kmeans

        self.C = torch.from_numpy(self.C_np)  # > comment it for sim kmeans
        print(f"size of C of kmeans model: {self.C.shape}")  # just for debugging (1024, 1000)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)  # > comment it for sim kmeans
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
        km_path = os.path.join(
            ckpt_dir,
            # "LibriSpeech_wavlm_k1000_L12.pt",
            "LibriSpeech100-360-500_wavlm_k1000_L6.pt",
            # "LibriSpeech100-360-500_wavlm_k2000_L7.pt",
            # "LibriSpeech_2048D_no_trim_batch_16384.pt",
            # "LibriSpeech_256D_no_trim_batch_16384.pt",
            # "LibriSpeech_1024D_no_trim_batch_16384.pt",
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
            ]  # (batch_size, sequence_length, 1024)
            features = layer_output.flatten(
                end_dim=-2
            )  # (sequence_length, hidden_size)

        return features.cpu().numpy()

    def get_cluster_embedding(self, cluster_id):
        # Get the cluster center (embedding) for a given cluster ID
        return self.kmeans.C_np[:, cluster_id]

    def __call__(self, path, merged=True, downsample=False, residual=False, save_residuals=False):
        if residual:
            residual_path = str(path).replace(
                "EmotionSpeechDataset", "EmotionSpeechDataset_residual_6L-1000k"
                # "data",
                # "data_residual_6L-1000k",
                # "audio", "audio_residual_6L-1000k",
            )
            residual_path = os.path.splitext(residual_path)[0] + ".npy"
            residual_np = np.load(residual_path)
            residual_length = residual_np.shape[0]

        try:
            features = self.extract_wavlm_features(path)
        except RuntimeError as e:
            raise e

        cluster_ids = self.kmeans(features).tolist()

        if save_residuals:
            cluster_embeddings = np.stack([self.get_cluster_embedding(idx) for idx in cluster_ids]) # (length, 1024)
            residual_features = cluster_embeddings - features
            residual_path = str(path).replace("data", "data_residual_6L-1000k_new")
            residual_path = os.path.splitext(residual_path)[0] + ".npy"
            os.makedirs(os.path.dirname(residual_path), exist_ok=True)
            print(f"shape of residual: {residual_features.shape}")
            print(f"save path: {residual_path}")
            np.save(residual_path, residual_features)

        if merged:
            new_cluster_list, duration_list = self.merge_duplicates(cluster_ids)
        elif downsample:
            new_cluster_list, duration_list = self.downsample(cluster_ids)
        else:
            new_cluster_list = cluster_ids

        # merged_units = (
        #     "<sosp>" + "".join([f"<{str(x)}>" for x in dup_cluster_list]) + "<eosp>"
        # )
        units = "<sosp>" + "".join([f"<{str(x)}>" for x in new_cluster_list]) + "<eosp>"
        # unmerged_units = (
        #     "<sosp>" + "".join([f"<{str(x)}>" for x in cluster_ids]) + "<eosp>"
        # )

        # if merged:
        #     return merged_units
        # else:
        #     return unmerged_units
        if residual:
            return units, residual_length, residual_path
        else:
            return units


if __name__ == "__main__":
    # raise RuntimeError("Direct execution is not currently supported (˘･_･˘)")
    s2u = Speech2UnitCustom("/home/jhwan98/EmoSDS/utils/speech2unit")

    speech_path = "/shared/NAS_SSD/jhl/futureinternet/EmotionSpeechDataset/0020/Surprise/0020_001750.wav"
    # speech1_path = "/home/jhwan98/0016_000724.wav"
    # speech2_path = "/home/jhwan98/0016_000374.wav"
    # speech3_path = "/home/jhwan98/0016_001424.wav"

    unit = s2u(speech_path, merged=False)
    # unit1 = s2u(speech1_path, merged=False)
    # unit2 = s2u(speech2_path, merged=False)
    # unit3 = s2u(speech3_path, merged=False)

    # with open("/home/jhwan98/units_layer6_k256_no_merge.txt", "w") as f:
    #     f.write(f"0016_000724\t{unit1}\n")
    #     f.write(f"0016_000374\t{unit2}\n")
    #     f.write(f"0016_001424\t{unit3}\n")

    # print("Successfully saved the units")

# python3 utils/speech2unit/speech2unit.py
