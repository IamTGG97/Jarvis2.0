"""Benchmark small local vision-language models (VLMs) through Ollama.

Usage (Ollama running; nothing else needed):

    .\\.venv\\Scripts\\python.exe backend\\bench_vlm.py
    .\\.venv\\Scripts\\python.exe backend\\bench_vlm.py --models moondream --json out.json

Image set (all license-clean, nothing is written to disk):
- ``office_public_domain.jpg`` (public domain, FEMA; see test_data/README.md)
  and a crop of it.
- Synthetic images drawn in memory with Pillow by this script (shapes, signs,
  text). They are generated fresh on every run.

Expected answers are written down in ``CASES`` before any model is run. A case
passes when the answer contains one of the ``expect`` keywords (whole word,
case-insensitive) and none of the ``reject`` keywords. Raw answers are printed
so a person can double-check the automatic score.

For each model: the model is unloaded first, the first question is the "cold"
call (includes model load), the rest are "warm" (model kept loaded with a short
keep_alive), then the model is unloaded again. A background thread samples the
Ollama processes' memory and the system's available RAM every 0.5 s.
"""

from __future__ import annotations

import argparse
import base64
import csv
import ctypes
import io
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

TEST_DATA = Path(__file__).resolve().parent / "test_data"
OFFICE = TEST_DATA / "office_public_domain.jpg"
OLLAMA = "http://127.0.0.1:11434"


# ---------------------------------------------------------------- images ---

def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (640, 480), "white")
    return image, ImageDraw.Draw(image)


def blue_circle() -> Image.Image:
    image, draw = _canvas()
    draw.ellipse((200, 120, 440, 360), fill=(30, 70, 220))
    return image


def text_sign() -> Image.Image:
    image, draw = _canvas()
    draw.rectangle((60, 150, 580, 330), fill=(250, 250, 240), outline="black", width=6)
    draw.text((320, 240), "ACME COFFEE", fill="black", font=_font(64), anchor="mm")
    return image


def stop_sign() -> Image.Image:
    image, draw = _canvas()
    cx, cy, r = 320, 240, 170
    import math

    points = [
        (cx + r * math.cos(math.radians(22.5 + 45 * i)), cy + r * math.sin(math.radians(22.5 + 45 * i)))
        for i in range(8)
    ]
    draw.polygon(points, fill=(200, 20, 30))
    draw.text((cx, cy), "STOP", fill="white", font=_font(90), anchor="mm")
    return image


def three_red_squares() -> Image.Image:
    image, draw = _canvas()
    for x in (70, 260, 450):
        draw.rectangle((x, 170, x + 120, 290), fill=(210, 30, 30))
    return image


def five_blue_dots() -> Image.Image:
    image, draw = _canvas()
    for x, y in ((90, 120), (300, 90), (500, 140), (180, 330), (430, 320)):
        draw.ellipse((x, y, x + 80, y + 80), fill=(30, 70, 220))
    return image


def big_green_small_yellow() -> Image.Image:
    image, draw = _canvas()
    draw.rectangle((40, 90, 340, 390), fill=(30, 160, 60))
    draw.rectangle((450, 200, 530, 280), fill=(240, 210, 20))
    return image


def office() -> Image.Image:
    return Image.open(OFFICE).convert("RGB")


def office_bottle_crop() -> Image.Image:
    crop = office().crop((190, 330, 310, 515))  # the water bottle on the table
    return crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)


def jpeg_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ------------------------------------------------------------------ cases ---

@dataclass
class Case:
    id: str
    category: str
    image: callable
    question: str
    expect: list[str]
    reject: list[str] = field(default_factory=list)


# Expected answers written before running any model.
CASES = [
    Case("color-circle", "color", blue_circle, "What color is the circle?", ["blue"]),
    Case("text-sign", "text/brand", text_sign, "What does the sign say?", ["acme"]),
    Case("text-stop", "text/brand", stop_sign, "What word is written on the sign?", ["stop"]),
    Case("count-3", "counting", three_red_squares, "How many red squares are there?", ["3", "three"],
         ["2", "4", "two", "four"]),
    Case("count-5", "counting", five_blue_dots, "How many blue circles are in the image?", ["5", "five"],
         ["3", "4", "6", "three", "four", "six"]),
    Case("compare-size", "comparison", big_green_small_yellow,
         "Which rectangle is bigger, the green one or the yellow one?", ["green"],
         ["yellow one is bigger", "yellow rectangle is bigger", "yellow is bigger"]),
    Case("office-what", "what is this", office, "What is happening in this picture?",
         ["office", "meeting", "desk", "table", "laptop", "laptops", "computer", "computers", "working"]),
    Case("office-count", "counting", office, "How many people are in this picture?", ["3", "three"],
         ["2", "4", "two", "four"]),
    Case("office-shirt", "color", office,
         "What color is the shirt of the man with glasses on the right?", ["blue", "navy"]),
    Case("office-folder", "color", office,
         "What color is the folder in the bottom left corner?", ["red"]),
    Case("office-brand", "text/brand", office,
         "What brand is the black laptop on the left?", ["hp", "hewlett"]),
    Case("bottle-what", "what is this", office_bottle_crop, "What is this object?",
         ["bottle", "water"]),
]


def score(case: Case, answer: str) -> bool:
    text = answer.lower()

    def has(word: str) -> bool:
        return re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text) is not None

    return any(has(w) for w in case.expect) and not any(has(w) for w in case.reject)


# ----------------------------------------------------------------- memory ---

class _MemStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_ram_mb() -> float:
    status = _MemStatus()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 2**20


def ollama_pids() -> list[int]:
    out = subprocess.run(
        ["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, check=False
    ).stdout
    pids = []
    for row in csv.reader(out.splitlines()):
        if row and row[0].lower().startswith(("ollama", "llama")):
            pids.append(int(row[1]))
    return pids


class MemorySampler:
    """Peak working set / private bytes of the Ollama processes, min free RAM."""

    def __init__(self) -> None:
        from bench_vision import process_memory_mb  # reuse the Windows helper

        self._mem = process_memory_mb
        self.peak_ws = 0.0
        self.min_avail = float("inf")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        pids: list[int] = []
        last_scan = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_scan > 2:
                pids, last_scan = ollama_pids(), now
            total = 0.0
            for pid in pids:
                reading = self._mem(pid)
                if reading:
                    total += reading[0]
            self.peak_ws = max(self.peak_ws, total)
            self.min_avail = min(self.min_avail, available_ram_mb())
            self._stop.wait(0.5)

    def __enter__(self) -> "MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


# ------------------------------------------------------------------ ollama ---

def unload_all(client: httpx.Client) -> None:
    for model in client.get(f"{OLLAMA}/api/ps").json().get("models", []):
        client.post(f"{OLLAMA}/api/generate", json={"model": model["name"], "keep_alive": 0})
    time.sleep(1)


STYLES = {
    # The plain question, as the user asked it.
    "question": "{q}",
    # moondream answers "describe" requests far more reliably than terse
    # questions on this Ollama build; this is the backend's retry prompt.
    "describe": "Describe this image in one sentence, then answer: {q}",
}


def ask(client: httpx.Client, model: str, case: Case, keep_alive: str, num_ctx: int,
        style: str = "question") -> dict:
    """Same call shape as the backend (``/api/generate``, temperature 0)."""
    started = time.perf_counter()
    response = client.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": model,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": 0, "num_predict": 80, "num_ctx": num_ctx},
            "prompt": STYLES[style].format(q=case.question),
            "images": [jpeg_b64(case.image())],
        },
    )
    wall = time.perf_counter() - started
    if response.status_code != 200:
        raise ValueError(f"HTTP {response.status_code}: {response.text[:200]}")
    body = response.json()
    return {
        "answer": (body.get("response") or "").strip(),
        "wall_s": round(wall, 2),
        "load_s": round(body.get("load_duration", 0) / 1e9, 2),
        "prompt_tokens": body.get("prompt_eval_count"),
        "prompt_s": round(body.get("prompt_eval_duration", 0) / 1e9, 2),
        "eval_s": round(body.get("eval_duration", 0) / 1e9, 2),
    }


def run_model(model: str, num_ctx: int, cases: list[Case], style: str = "question",
              timeout: float = 900) -> dict:
    results = []
    with httpx.Client(timeout=timeout) as client:
        unload_all(client)
        before = available_ram_mb()
        with MemorySampler() as sampler:
            for index, case in enumerate(cases):
                keep = "0" if index == len(cases) - 1 else "5m"
                try:
                    result = ask(client, model, case, keep, num_ctx, style)
                except (httpx.HTTPError, ValueError) as error:
                    result = {"answer": f"ERROR: {error}", "wall_s": None, "load_s": None,
                              "prompt_tokens": None, "prompt_s": None, "eval_s": None}
                result.update(id=case.id, category=case.category, cold=index == 0,
                              correct=score(case, result["answer"]))
                results.append(result)
                print(f"  [{'cold' if index == 0 else 'warm'}] {case.id}: {result['wall_s']} s "
                      f"{'OK ' if result['correct'] else 'BAD'} | {result['answer'][:110]!r}", flush=True)
        unload_all(client)
    warm = [r["wall_s"] for r in results[1:] if r["wall_s"] is not None]
    return {
        "model": model,
        "style": style,
        "num_ctx": num_ctx,
        "available_before_mb": round(before),
        "min_available_mb": round(sampler.min_avail),
        "peak_ollama_working_set_mb": round(sampler.peak_ws),
        "cold_s": results[0]["wall_s"],
        "warm_median_s": round(sorted(warm)[len(warm) // 2], 2) if warm else None,
        "warm_max_s": max(warm) if warm else None,
        "correct": sum(r["correct"] for r in results),
        "total": len(results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local VLMs through Ollama")
    parser.add_argument("--models", nargs="+", default=["moondream", "granite3.2-vision"])
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--style", choices=sorted(STYLES), default="question")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per request")
    parser.add_argument("--cases", nargs="*", help="only these case ids")
    parser.add_argument("--json", type=Path, help="write full results here")
    args = parser.parse_args()

    cases = [c for c in CASES if not args.cases or c.id in args.cases]
    summary = []
    for model in args.models:
        print(f"== {model}", flush=True)
        summary.append(run_model(model, args.num_ctx, cases, args.style, args.timeout))
        time.sleep(3)

    print("\nmodel | correct | cold s | warm median s | warm max s | peak Ollama WS MB | min free RAM MB")
    for s in summary:
        print(f"{s['model']} | {s['correct']}/{s['total']} | {s['cold_s']} | {s['warm_median_s']} | "
              f"{s['warm_max_s']} | {s['peak_ollama_working_set_mb']} | {s['min_available_mb']}")
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
