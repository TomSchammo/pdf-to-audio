"""Compare Kokoro on PyTorch against Kokoro on ONNX Runtime (fp32 and int8) on CPU.

The text is converted to phonemes once, with the same misaki G2P that main.py
uses, and every engine synthesizes exactly those chunks. Differences in the
results therefore come from the model runtime alone. Each engine runs in its
own process so load time and peak memory are measured cleanly.

    uv run --group bench benchmark.py data/paper.txt
"""

import argparse
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ONNX Runtime sends usage telemetry to Microsoft and stores a device ID by default.
# Must be set before onnxruntime is imported; child processes inherit it.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SAMPLE_RATE = 24000
REPO_ID = "hexgrad/Kokoro-82M"
ONNX_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
ONNX_MODELS = {"onnx": "kokoro-v1.0.onnx", "onnx-int8": "kokoro-v1.0.int8.onnx"}
ONNX_VOICES = "voices-v1.0.bin"
ENGINES = ["torch", *ONNX_MODELS]

log = logging.getLogger("benchmark")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="text file to synthesize")
    parser.add_argument(
        "-e",
        "--engines",
        nargs="+",
        choices=ENGINES,
        default=ENGINES,
        help="engines to run (default: all)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        help="only use roughly the first N characters, cut at a line break",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=os.cpu_count(),
        help="CPU threads per engine (default: all cores)",
    )
    parser.add_argument("-s", "--speed", type=float, default=1.2)
    parser.add_argument("-v", "--voice", default="af_heart")
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/bench"))
    # Internal: run a single engine in a child process.
    parser.add_argument("--child", choices=ENGINES, help=argparse.SUPPRESS)
    return parser.parse_args()


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS but kilobytes on Linux.
    return rss / 1024**2 if sys.platform == "darwin" else rss / 1024


def read_text(path, max_chars):
    text = path.read_text(encoding="utf-8")
    if max_chars and len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        text = text[: cut if cut > 0 else max_chars]
    return text


def phonemize(text):
    """Split the text into phoneme chunks exactly as KPipeline does in main.py."""
    from kokoro import KPipeline

    start = time.perf_counter()
    pipeline = KPipeline(lang_code="a", repo_id=REPO_ID, model=False)
    load_time = time.perf_counter() - start

    start = time.perf_counter()
    chunks = [result.phonemes for result in pipeline(text)]
    return chunks, load_time, time.perf_counter() - start


def download_onnx_files(engines, models_dir):
    names = [ONNX_MODELS[e] for e in engines if e in ONNX_MODELS]
    if names:
        names.append(ONNX_VOICES)
    models_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = models_dir / name
        if not path.exists():
            log.info("Downloading %s", name)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(f"{ONNX_RELEASE}/{name}", tmp)
            tmp.rename(path)


def load_torch(args):
    """Return the import time, load time and a synthesize(phonemes) function."""
    start = time.perf_counter()
    import torch
    from huggingface_hub import hf_hub_download
    from kokoro import KModel

    import_time = time.perf_counter() - start

    start = time.perf_counter()
    torch.set_num_threads(args.threads)
    model = KModel(repo_id=REPO_ID).eval()
    voice_file = hf_hub_download(REPO_ID, f"voices/{args.voice}.pt")
    pack = torch.load(voice_file, weights_only=True)
    load_time = time.perf_counter() - start

    def synthesize(ps):
        with torch.inference_mode():
            return model(ps, pack[len(ps) - 1], args.speed).numpy()

    return import_time, load_time, synthesize


def load_onnx(args):
    start = time.perf_counter()
    import onnxruntime as rt
    from kokoro_onnx import Kokoro

    import_time = time.perf_counter() - start

    start = time.perf_counter()
    options = rt.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = rt.InferenceSession(
        str(args.models_dir / ONNX_MODELS[args.child]),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    kokoro = Kokoro.from_session(session, str(args.models_dir / ONNX_VOICES))
    load_time = time.perf_counter() - start

    def synthesize(ps):
        # No trimming or added pauses, so the output matches the PyTorch engine.
        audio, _ = kokoro.create(
            ps,
            args.voice,
            speed=args.speed,
            is_phonemes=True,
            trim=False,
            sentence_pause=0,
            clause_pause=0,
        )
        return audio

    return import_time, load_time, synthesize


def run_child(args):
    """Synthesize the prepared chunks with one engine and print the metrics as JSON."""
    chunks = json.loads((args.out_dir / "chunks.json").read_text(encoding="utf-8"))
    loader = load_torch if args.child == "torch" else load_onnx
    import_time, load_time, synthesize = loader(args)
    log.info("%s: imported in %.2fs, loaded in %.2fs", args.child, import_time, load_time)

    audio = []
    start = time.perf_counter()
    for i, ps in enumerate(chunks):
        audio.append(synthesize(ps))
        if i == 0:
            first_chunk = time.perf_counter() - start
        log.debug("%s: chunk %d/%d", args.child, i + 1, len(chunks))
    gen_time = time.perf_counter() - start

    merged = np.concatenate(audio)
    sf.write(args.out_dir / f"{args.child}.wav", merged, SAMPLE_RATE)
    audio_seconds = len(merged) / SAMPLE_RATE
    print(
        json.dumps(
            {
                "engine": args.child,
                "import_s": import_time,
                "load_s": load_time,
                "first_chunk_s": first_chunk,
                "generate_s": gen_time,
                "audio_s": audio_seconds,
                "rtf": audio_seconds / gen_time,
                "peak_rss_mb": peak_rss_mb(),
            }
        )
    )


def run_engine(engine, args):
    command = [
        sys.executable,
        __file__,
        str(args.input),
        "--child",
        engine,
        "--threads",
        str(args.threads),
        "--speed",
        str(args.speed),
        "--voice",
        args.voice,
        "--models-dir",
        str(args.models_dir),
        "--out-dir",
        str(args.out_dir),
    ]
    # The child logs to stderr, which passes through; its last stdout line is the result.
    done = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def print_table(results):
    columns = [
        ("engine", "engine", "{}"),
        ("import_s", "import", "{:.2f}s"),
        ("load_s", "load", "{:.2f}s"),
        ("first_chunk_s", "1st chunk", "{:.2f}s"),
        ("generate_s", "generate", "{:.1f}s"),
        ("rtf", "real-time", "{:.1f}x"),
        ("peak_rss_mb", "peak RSS", "{:.0f} MB"),
    ]
    rows = [[fmt.format(r[key]) for key, _, fmt in columns] for r in results]
    headers = [title for _, title, _ in columns]
    widths = [max(len(c) for c in col) for col in zip(headers, *rows)]
    for row in [headers, ["-" * w for w in widths], *rows]:
        print("  ".join(cell.rjust(w) for cell, w in zip(row, widths)))


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
    )
    log.setLevel(logging.INFO)

    if args.child:
        run_child(args)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    text = read_text(args.input, args.max_chars)
    chunks, g2p_load, g2p_time = phonemize(text)
    (args.out_dir / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    log.info(
        "G2P: %d characters -> %d chunks in %.2fs (setup %.2fs)",
        len(text),
        len(chunks),
        g2p_time,
        g2p_load,
    )

    download_onnx_files(args.engines, args.models_dir)

    results = []
    for engine in args.engines:
        log.info("Running %s with %d threads", engine, args.threads)
        results.append(run_engine(engine, args))

    print()
    print_table(results)

    report = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "input": {"file": str(args.input), "chars": len(text), "chunks": len(chunks)},
        "settings": {"threads": args.threads, "speed": args.speed, "voice": args.voice},
        "g2p": {"setup_s": g2p_load, "phonemize_s": g2p_time},
        "results": results,
    }
    report_path = args.out_dir / "results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nAudio and {report_path} written to {args.out_dir}/")


if __name__ == "__main__":
    main()
