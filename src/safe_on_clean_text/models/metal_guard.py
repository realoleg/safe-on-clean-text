from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from safe_on_clean_text.provenance import timestamp

# Any Metal command-buffer error (out of memory or other): PyTorch does not raise on it, the failed command buffer
# is silently skipped, so a run that printed one is not trustworthy.
METAL_ERROR = b"kIOGPUCommandBufferCallbackError"


def run_folder_complete(run_dir: Path) -> bool:
    """
    A run is complete when its run_meta.json (written last) has the best-checkpoint check.
    Runs from before the LayerNorm fix lack it and are trained again.
    """

    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return False
    with open(meta_path, encoding="utf-8") as f:
        return "best_checkpoint_check" in json.load(f)


def run_guarded(command: list[str]) -> int:
    """
    Run a command, passing its combined output through to stdout. The process is killed at the first Metal
    command-buffer error. Returns the number of Metal errors seen; raises on a non-zero exit without them.
    """

    print(f"[{timestamp()}] RUN {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    n_errors = 0
    tail = b""
    for chunk in iter(lambda: process.stdout.read1(65536), b""):
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        # The kept tail is shorter than the pattern, so a match is never counted twice.
        window = tail + chunk
        n_errors += window.count(METAL_ERROR)
        tail = window[-(len(METAL_ERROR) - 1) :]
        if n_errors and process.poll() is None:
            process.kill()
    process.wait()

    if n_errors:
        print(f"[{timestamp()}] METAL ERROR in {' '.join(command)}: process killed", flush=True)
        return n_errors
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    print(f"[{timestamp()}] DONE {' '.join(command)}", flush=True)
    return 0


def swap_usage() -> str:
    return subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
