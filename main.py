import argparse
import ctypes
import logging
import os
import resource
import struct
import sys
import time
from pathlib import Path

# Must be set before torch is imported (kokoro imports it) to take effect.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from kokoro import KPipeline  # noqa: E402
from kokoro.istftnet import TorchSTFT  # noqa: E402

SAMPLE_RATE = 24000
REPO_ID = "hexgrad/Kokoro-82M"

log = logging.getLogger("pdf-to-audio")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a text file to speech.")
    parser.add_argument("input", type=Path, help="text file to read")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output .wav file (default: input name with .wav extension)",
    )
    parser.add_argument(
        "-s", "--speed", type=float, default=1.2, help="speech speed (default: 1.2)"
    )
    parser.add_argument(
        "-v", "--voice", default="af_heart", help="voice name (default: af_heart)"
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="device to run the model on (default: auto)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="log text and timing for every chunk"
    )
    return parser.parse_args()


def pick_device(requested):
    # Kokoro's own auto-selection only knows about CUDA, so MPS must be passed explicitly.
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CPUSTFT(torch.nn.Module):
    """Kokoro's STFT, run on the CPU while the rest of the model stays on the GPU.

    PyTorch's MPS STFT makes the audio sound metallic; on the CPU the output
    matches a pure CPU run, at about the same speed as all-MPS.
    """

    def __init__(self, stft):
        super().__init__()
        self.stft = stft.cpu()

    def transform(self, x):
        magnitude, phase = self.stft.transform(x.cpu())
        return magnitude.to(x.device), phase.to(x.device)

    def inverse(self, magnitude, phase):
        audio = self.stft.inverse(magnitude.cpu(), phase.cpu())
        return audio.to(magnitude.device)


def move_stft_to_cpu(model):
    generator = model.decoder.generator
    # Reaches into Kokoro internals; fail loudly if a new version changes them.
    if not isinstance(getattr(generator, "stft", None), TorchSTFT):
        raise RuntimeError(
            "Kokoro's vocoder layout changed; cannot apply the MPS STFT fix. "
            "Use --device cpu."
        )
    generator.stft = CPUSTFT(generator.stft)


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS but kilobytes on Linux.
    return rss / 1024**2 if sys.platform == "darwin" else rss / 1024


def footprint_mb():
    """Physical footprint, the memory figure Activity Monitor shows. macOS only.

    Unlike RSS it includes GPU allocations, which share RAM on Apple Silicon.
    """
    if sys.platform != "darwin":
        return None
    info = ctypes.create_string_buffer(256)
    libproc = ctypes.CDLL("libproc.dylib")
    if libproc.proc_pid_rusage(os.getpid(), 0, info) != 0:  # RUSAGE_INFO_V0
        return None
    # rusage_info_v0: a 16-byte UUID, then uint64 fields; ri_phys_footprint is the 8th.
    return struct.unpack_from("Q", info, 16 + 7 * 8)[0] / 1024**2


def gpu_memory_mb(device):
    """GPU memory in use, or None on CPU. RSS does not include it."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    if device.type == "mps":
        # MPS has no peak counter, so this is sampled after every chunk.
        return torch.mps.driver_allocated_memory() / 1024**2
    return None


def main():
    args = parse_args()
    # Third-party libraries only log warnings; our own logger is more verbose.
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
    )
    log.setLevel(logging.DEBUG if args.debug else logging.INFO)

    output = args.output or args.input.with_suffix(".wav")
    text = args.input.read_text(encoding="utf-8")
    log.info("Read %d characters from %s", len(text), args.input)

    start = time.perf_counter()
    pipeline = KPipeline(lang_code="a", repo_id=REPO_ID, device=pick_device(args.device))
    device = pipeline.model.device
    if device.type == "mps":
        move_stft_to_cpu(pipeline.model)
    load_time = time.perf_counter() - start
    log.info("Loaded model on %s in %.2fs", device, load_time)

    start = time.perf_counter()
    chunks = []
    peak_gpu = gpu_memory_mb(device)
    peak_footprint = footprint_mb()
    chunk_start = start
    for i, (gs, ps, audio) in enumerate(
        pipeline(text, voice=args.voice, speed=args.speed)
    ):
        now = time.perf_counter()
        chunk_time, chunk_start = now - chunk_start, now
        log.debug(
            "Chunk %d: %.2fs for %.2fs audio | %s | %s",
            i,
            chunk_time,
            len(audio) / SAMPLE_RATE,
            gs,
            ps,
        )
        chunks.append(audio)
        if device.type == "mps":
            # Otherwise PyTorch keeps new blocks for every chunk length and
            # memory grows by ~35 MB per chunk.
            torch.mps.empty_cache()
        if peak_gpu is not None:
            peak_gpu = max(peak_gpu, gpu_memory_mb(device))
        if peak_footprint is not None:
            footprint = footprint_mb()
            peak_footprint = max(peak_footprint, footprint)
            if (i + 1) % 25 == 0:
                log.info("After chunk %d: memory footprint %.0f MB", i + 1, footprint)
    gen_time = time.perf_counter() - start

    if not chunks:
        log.error("No audio generated from %s", args.input)
        sys.exit(1)

    start = time.perf_counter()
    merged = np.concatenate(chunks)
    sf.write(output, merged, SAMPLE_RATE)
    write_time = time.perf_counter() - start

    audio_seconds = len(merged) / SAMPLE_RATE
    log.info("Wrote %s (%.1fs audio, %d chunks)", output, audio_seconds, len(chunks))
    log.info(
        "Timing: load %.2fs, generate %.2fs, write %.2fs | "
        "real-time factor %.1fx | %.0f chars/s | peak RSS %.0f MB",
        load_time,
        gen_time,
        write_time,
        audio_seconds / gen_time,
        len(text) / gen_time,
        peak_rss_mb(),
    )
    if peak_gpu is not None:
        log.info("Peak %s memory %.0f MB", device.type.upper(), peak_gpu)
    if peak_footprint is not None:
        log.info("Peak memory footprint %.0f MB (as in Activity Monitor)", peak_footprint)


if __name__ == "__main__":
    main()
