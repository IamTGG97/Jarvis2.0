"""Benchmark POST /api/detect against a running backend.

Usage (backend already running on 127.0.0.1:8000):

    .\\.venv\\Scripts\\python.exe backend\\bench_vision.py
    .\\.venv\\Scripts\\python.exe backend\\bench_vision.py --runs 20 --image path\\to\\photo.jpg --pid 1234

Reports end-to-end request latency (first call = cold, includes model load) and
the server-reported inference_ms. With --pid (the uvicorn python process) it
also prints that process's working set and peak working set on Windows.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import sys
import time
from pathlib import Path

import httpx

DEFAULT_IMAGE = Path(__file__).resolve().parent / "test_data" / "office_public_domain.jpg"


def process_memory_mb(pid: int) -> tuple[float, float] | None:
    """(working set, peak working set) in MB for a Windows process, else None."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED_INFORMATION | VM_READ
    if not handle:
        return None
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return counters.WorkingSetSize / 2**20, counters.PeakWorkingSetSize / 2**20
    finally:
        kernel32.CloseHandle(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark POST /api/detect")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/detect")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--pid", type=int, help="uvicorn process id for memory readings")
    args = parser.parse_args()

    data = args.image.read_bytes()
    content_type = "image/png" if args.image.suffix.lower() == ".png" else "image/jpeg"

    def mem(label: str) -> None:
        if args.pid:
            reading = process_memory_mb(args.pid)
            if reading:
                print(f"  {label}: working set {reading[0]:.0f} MB, peak {reading[1]:.0f} MB")

    mem("before first request")
    walls, server = [], []
    with httpx.Client(timeout=120) as client:
        for i in range(args.runs + 1):
            started = time.perf_counter()
            response = client.post(args.url, files={"file": (args.image.name, data, content_type)})
            wall = (time.perf_counter() - started) * 1000
            if response.status_code != 200:
                print(f"HTTP {response.status_code}: {response.text}")
                return 1
            body = response.json()
            if i == 0:
                labels = ", ".join(f"{d['label']} {d['confidence']:.2f}" for d in body["detections"])
                print(f"cold: {wall:.0f} ms wall, {body['inference_ms']:.0f} ms inference")
                print(f"  person_present={body['person_present']}; {labels}")
                mem("after cold request")
            else:
                walls.append(wall)
                server.append(body["inference_ms"])
    print(
        f"warm x{args.runs}: wall median {statistics.median(walls):.0f} ms "
        f"(min {min(walls):.0f}, max {max(walls):.0f}); "
        f"inference median {statistics.median(server):.0f} ms"
    )
    mem("after warm requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
