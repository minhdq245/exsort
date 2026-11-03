"""One writer per output directory; validate PID creation time to reject PID reuse."""
from contextlib import contextmanager
import os
from pathlib import Path
import psutil
from .io import read, save


@contextmanager
def writer_lock(folder):
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    lock = root/".writer.lock"
    token = dict(pid=os.getpid(), created=psutil.Process().create_time())
    if lock.exists():
        old = read(lock)
        try:
            process = psutil.Process(old["pid"])
            active = abs(process.create_time()-old["created"]) < .01
        except psutil.NoSuchProcess:
            active = False
        if active:
            raise RuntimeError(f"Output is already owned by a live process: {old['pid']}")
        lock.unlink()
    # Create lock contents in one exclusive open; never replace a live writer's lock.
    import json
    with lock.open("x", encoding="utf8") as f:
        json.dump(token, f)
    try:
        yield
    finally:
        if lock.exists() and read(lock) == token:
            lock.unlink()
