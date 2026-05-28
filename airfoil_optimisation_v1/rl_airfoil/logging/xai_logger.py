from __future__ import annotations
from pathlib import Path
import csv
from typing import Dict, List


class CSVLogger:
    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.w.writeheader()

    def log(self, row: Dict):
        safe = {k: row.get(k, "") for k in self.fieldnames}
        self.w.writerow(safe)

    def close(self):
        self.f.close()
