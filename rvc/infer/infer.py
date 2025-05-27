import os
import sys
import soxr
import time
import torch
import librosa
import logging
import traceback
import numpy as np
import soundfile as sf
import noisereduce as nr

# ANSI color codes for console output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

now_dir = os.getcwd()
sys.path.append(os.getcwd())

from rvc.infer.pipeline import Pipeline as VC
from rvc.lib.utils import load_audio_infer, load_embedding
from rvc.lib.tools.split_audio import process_audio, merge_audio
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.configs.config import Config

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)


class VoiceConverter:
    """
    A class for performing voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self):
        """
        Initializes the VoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()
        self.hubert_model = None
        self.last_embedder_model = None
        self.tgt_sr = None
        self.net_g = None
        self.vc = None
        self.cpt = None
        self.version = None
        self.n_spk = None
        self.use_f0 = None
        self.loaded_model = None

    def load_hubert(self, embedder_model: str, embedder_model_custom: str = None):
        """
        Loads the HuBERT model for speaker embedding extraction.

        Args:
            embedder_model (str): Path to the pre-trained HuBERT model.
            embedder_model_custom (str): Path to the custom HuBERT model.
        """
        print(f"{Colors.YELLOW}Loading HuBERT model for speaker embedding...{Colors.RESET}")
        self.hubert_model = load_embedding(embedder_model, embedder_model_custom)
        self.hubert_model = self.hubert_model.to(self.config.device).float()
        self.hubert_model.eval()
        print(f"{Colors.GREEN}HuBERT model loaded successfully.{Colors.RESET}")

    @staticmethod
    def remove_audio_noise(data, sr, reduction_strength=0.7):
        """
        Removes noise from an audio file using the NoiseReduce library.

        Args:
            data (numpy.ndarray): The audio data as a NumPy array.
            sr (int): The sample rate of the audio data.
            reduction_strength (float): Strength of the noise reduction. Default is 0.7.
        """
        try:
            print(f"{Colors.YELLOW}Applying noise reduction (strength: {reduction_strength})...{Colors.RESET}")
            reduced_noise = nr.reduce_noise(
                y=data, sr=sr, prop_decrease=reduction_strength
            )
            print(f"{Colors.GREEN}Noise reduction applied successfully.{Colors.RESET}")
            return reduced_noise
        except Exception as error:
            print(f"{Colors.RED}Error during noise reduction: {error}{Colors.RESET}")
            return None

    @staticmethod
    def convert_audio_format(input_path, output_path, output_format):
        """
        Converts an audio file to a specified output format.

        Args:
            input_path (str): Path to the input audio file.
            output_path (str): Path to the output audio file.
            output_format (str): Desired audio format (e.g., "WAV", "MP3").
        """
        try:
            if output_format != "WAV":
                print(f"{Colors.YELLOW}Converting audio to {output_format} format...{Colors.RESET}")
                audio, sample_rate = librosa.load(input_path, sr=None)
                common_sample_rates = [
                    8000,
                    11025,
                    12000,
                    16000,
                    22050,
                    24000,
                    32000,
                    44100,
                    48000,
                ]
                target_sr = min(common_sample_rates, key=lambda x: abs(x - sample_rate))
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=target_sr, res_type="soxr_vhq"
                )
                sf.write(output_path, audio, target_sr, format=output_format.lower())
                print(f"{Colors.GREEN}Audio converted to {output_format} and saved at '{output_path}'.{Colors.RESET}")
            return output_path
        except Exception as error:
            print(f"{Colors.RED}Error converting audio format: {error}{Colors.RESET}")

    def convert_audio(
        self,
        audio_input_path: str,
        audio_output_path: str,
        model_path: str,
        index_path: str,
        pitch: int = 0,
        f0_file: str = None,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        volume_envelope: float = 1,
        protect: float = 0.5,
        hop_length: int = 128,
        split_audio: bool = False,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        embedder_model: str = "contentvec",
        embedder_model_custom: str = None,
        clean_audio: bool = False,
        clean_strength: float = 0.5,
        export_format: str = "WAV",
        resample_sr: int = 0,
        sid: int = 0,
        **kwargs,
    ):
        """
        Performs voice conversion on the input audio.

        Args:
            pitch (int): Key for F0 up-sampling.
            index_rate (float): Rate for index matching.
            volume_envelope (int): RMS mix rate.
            protect (float): Protection rate for certain audio segments.
            hop_length (int): Hop length for audio processing.
            f0_method (str): Method for F0 extraction.
            audio_input_path (str): Path to the input audio file.
            audio_output_path (str): Path to the output audio file.
            model_path (str): Path to the voice conversion model.
            index_path (str): Path to the index file.
            split_audio (bool): Whether to split the audio for processing.
            f0_autotune (bool): Whether to use F0 autotune.
            clean_audio (bool): Whether to clean the audio.
            clean_strength (float): Strength of the audio cleaning.
            export_format (str): Format for exporting the audio.
            f0_file (str): Path to the F0 file.
            embedder_model (str): Path to the embedder model.
            embedder_model_custom (str): Path to the custom embedder model.
            resample_sr (int, optional): Resample sampling rate. Default is 0.
            sid (int, optional): Speaker ID. Default is 0.
            **kwargs: Additional keyword arguments.
        """
        if not model_path:
            print(f"{Colors.RED}No model path provided. Voice conversion aborted.{Colors.RESET}")
            return

        self.get_vc(model_path, sid)

        try:
            start_time = time.time()
            print(f"{Colors.YELLOW}Starting voice conversion for '{audio_input_path}'...{Colors.RESET}")
            print(f"{Colors.YELLOW}Using F0 method: {f0_method} with hop length: {hop_length}{Colors.RESET}")
            audio = load_audio_infer(
                audio_input_path,
                16000,
                **kwargs,
            )
            audio_max = np.abs(audio).max() / 0.95

            if audio_max > 1:
                print(f"{Colors.YELLOW}Normalizing audio to prevent clipping...{Colors.RESET}")
                audio /= audio_max

            if not self.hubert_model or embedder_model != self.last_embedder_model:
                self.load_hubert(embedder_model, embedder_model_custom)
                self.last_embedder_model = embedder_model

            file_index = (
                index_path.strip()
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip()
                .replace("trained", "added")
            )

            if self.tgt_sr != resample_sr >= 16000:
                self.tgt_sr = resample_sr

            if split_audio:
                print(f"{Colors.YELLOW}Splitting audio for processing...{Colors.RESET}")
                chunks, intervals = process_audio(audio, 16000)
                print(f"{Colors.GREEN}Audio split into {len(chunks)} chunks.{Colors.RESET}")
            else:
                chunks = [audio]

            converted_chunks = []
            for i, c in enumerate(chunks, 1):
                print(f"{Colors.YELLOW}Converting chunk {i}/{len(chunks)}...{Colors.RESET}")
                audio_opt = self.vc.pipeline(
                    model=self.hubert_model,
                    net_g=self.net_g,
                    sid=sid,
                    audio=c,
                    pitch=pitch,
                    f0_method=f0_method,
                    file_index=file_index,
                    index_rate=index_rate,
                    pitch_guidance=self.use_f0,
                    volume_envelope=volume_envelope,
                    version=self.version,
                    protect=protect,
                    hop_length=hop_length,
                    f0_autotune=f0_autotune,
                    f0_autotune_strength=f0_autotune_strength,
                    f0_file=f0_file,
                )
                converted_chunks.append(audio_opt)
                if split_audio:
                    print(f"{Colors.GREEN}Chunk {i}/{len(chunks)} converted successfully.{Colors.RESET}")

            if split_audio:
                print(f"{Colors.YELLOW}Merging {len(converted_chunks)} converted chunks...{Colors.RESET}")
                audio_opt = merge_audio(
                    chunks, converted_chunks, intervals, 16000, self.tgt_sr
                )
            else:
                audio_opt = converted_chunks[0]

            if clean_audio:
                print(f"{Colors.YELLOW}Cleaning audio with strength {clean_strength}...{Colors.RESET}")
                cleaned_audio = self.remove_audio_noise(
                    audio_opt, self.tgt_sr, clean_strength
                )
                if cleaned_audio is not None:
                    audio_opt = cleaned_audio

            print(f"{Colors.YELLOW}Saving converted audio to '{audio_output_path}'...{Colors.RESET}")
            sf.write(audio_output_path, audio_opt, self.tgt_sr, format="WAV")
            output_path_format = audio_output_path.replace(
                ".wav", f".{export_format.lower()}"
            )
            audio_output_path = self.convert_audio_format(
                audio_output_path, output_path_format, export_format
            )

            elapsed_time = time.time() - start_time
            print(
                f"{Colors.GREEN}Voice conversion completed successfully at '{audio_output_path}' in {elapsed_time:.2f} seconds.{Colors.RESET}"
            )
        except Exception as error:
            print(f"{Colors.RED}Error during voice conversion: {error}{Colors.RESET}")
            print(f"{Colors.RED}Traceback: {traceback.format_exc()}{Colors.RESET}")

    def convert_audio_batch(
        self,
        audio_input_paths: str,
        audio_output_path: str,
        **kwargs,
    ):
        """
        Performs voice conversion on a batch of input audio files.

        Args:
            audio_input_paths (str): List of paths to the input audio files.
            audio_output_path (str): Path to the output audio file.
            resample_sr (int, optional): Resample sampling rate. Default is 0.
            sid (int, optional): Speaker ID. Default is 0.
            **kwargs: Additional keyword arguments.
        """
        pid = os.getpid()
        try:
            with open(
                os.path.join(now_dir, "assets", "infer_pid.txt"), "w"
            ) as pid_file:
                pid_file.write(str(pid))
            start_time = time.time()
            print(f"{Colors.YELLOW}Starting batch voice conversion for directory '{audio_input_paths}'...{Colors.RESET}")
            audio_files = [
                f
                for f in os.listdir(audio_input_paths)
                if f.endswith(
                    (
                        "wav",
                        "mp3",
                        "flac",
                        "ogg",
                        "opus",
                        "m4a",
                        "mp4",
                        "aac",
                        "alac",
                        "wma",
                        "aiff",
                        "webm",
                        "ac3",
                    )
                )
            ]
            print(f"{Colors.GREEN}Found {len(audio_files)} audio files for processing.{Colors.RESET}")
            for i, a in enumerate(audio_files, 1):
                new_input = os.path.join(audio_input_paths, a)
                new_output = os.path.splitext(a)[0] + "_output.wav"
                new_output = os.path.join(audio_output_path, new_output)
                if os.path.exists(new_output):
                    print(f"{Colors.YELLOW}Skipping '{new_input}' (output already exists).{Colors.RESET}")
                    continue
                print(f"{Colors.YELLOW}Processing file {i}/{len(audio_files)}: '{new_input}'...{Colors.RESET}")
                self.convert_audio(
                    audio_input_path=new_input,
                    audio_output_path=new_output,
                    **kwargs,
                )
            print(f"{Colors.GREEN}Batch conversion completed for '{audio_input_paths}'.{Colors.RESET}")
            elapsed_time = time.time() - start_time
            print(f"{Colors.GREEN}Batch processing finished in {elapsed_time:.2f} seconds.{Colors.RESET}")
        except Exception as error:
            print(f"{Colors.RED}Error during batch conversion: {error}{Colors.RESET}")
            print(f"{Colors.RED}Traceback: {traceback.format_exc()}{Colors.RESET}")
        finally:
            os.remove(os.path.join(now_dir, "assets", "infer_pid.txt"))

    def get_vc(self, weight_root, sid):
        """
        Loads the voice conversion model and sets up the pipeline.

        Args:
            weight_root (str): Path to the model weights.
            sid (int): Speaker ID.
        """
        if sid == "" or sid == []:
            print(f"{Colors.YELLOW}No speaker ID provided. Cleaning up model...{Colors.RESET}")
            self.cleanup_model()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not self.loaded_model or self.loaded_model != weight_root:
            print(f"{Colors.YELLOW}Loading voice conversion model from '{weight_root}'...{Colors.RESET}")
            self.load_model(weight_root)
            if self.cpt is not None:
                self.setup_network()
                self.setup_vc_instance()
                print(f"{Colors.GREEN}Voice conversion model loaded successfully.{Colors.RESET}")
            self.loaded_model = weight_root

    def cleanup_model(self):
        """
        Cleans up the model and releases resources.
        """
        print(f"{Colors.YELLOW}Cleaning up model resources...{Colors.RESET}")
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del self.net_g, self.cpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cpt = None
        print(f"{Colors.GREEN}Model resources cleaned up.{Colors.RESET}")

    def load_model(self, weight_root):
        """
        Loads the model weights from the specified path.

        Args:
            weight_root (str): Path to the model weights.
        """
        print(f"{Colors.YELLOW}Loading model weights from '{weight

_root}'...{Colors.RESET}")
        self.cpt = (
            torch.load(weight_root, map_location="cpu", weights_only=True)
            if os.path.isfile(weight_root)
            else None
        )
        if self.cpt is None:
            print(f"{Colors.RED}Failed to load model weights from '{weight_root}'.{Colors.RESET}")
        else:
            print(f"{Colors.GREEN}Model weights loaded successfully.{Colors.RESET}")

    def setup_network(self):
        """
        Sets up the network configuration based on the loaded checkpoint.
        """
        if self.cpt is not None:
            print(f"{Colors.YELLOW}Setting up network configuration...{Colors.RESET}")
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
            self.use_f0 = self.cpt.get("f0", 1)
            self.version = self.cpt.get("version", "v1")
            self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
            self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")
            self.net_g = Synthesizer(
                *self.cpt["config"],
                use_f0=self.use_f0,
                text_enc_hidden_dim=self.text_enc_hidden_dim,
                vocoder=self.vocoder,
            )
            del self.net_g.enc_q
            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g = self.net_g.to(self.config.device).float()
            self.net_g.eval()
            print(f"{Colors.GREEN}Network configuration set up successfully.{Colors.RESET}")

    def setup_vc_instance(self):
        """
        Sets up the voice conversion pipeline instance based on the target sampling rate and configuration.
        """
        if self.cpt is not None:
            print(f"{Colors.YELLOW}Setting up voice conversion pipeline...{Colors.RESET}")
            self.vc = VC(self.tgt_sr, self.config)
            self.n_spk = self.cpt["config"][-3]
            print(f"{Colors.GREEN}Voice conversion pipeline initialized.{Colors.RESET}")
