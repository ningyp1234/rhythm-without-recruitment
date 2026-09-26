"""Range-fetch the three other published full-MANC DNg100 neural runs.

The publication's Figure 4 combines four run IDs into 128 parameter rows.
Run 33195882 was independently audited already; this adds the remaining 96
author-saved trajectories without downloading the 2.03 GB outer ZIP.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from fetch_pugliese_fullmanc_subset import ARCHIVE_SIZE, ROOT, URL, fetch_stored
from fetch_pugliese_zenodo_subset import HTTPRangeReader


RUNS = (30662613, 33195857, 33195876)
BASE = "simulations/DNg100_Stim_fullManc/hyak"
MEMBERS = ("ckpt/DNg100_Stim_fullManc_Rs.npz", "ckpt/neuron_params.h5",
           "logs/run_config.yaml", ".hydra/config.yaml", ".hydra/overrides.yaml")
OUT = ROOT / "data/pugliese-zenodo-22260924/full-manc-all-runs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reader = HTTPRangeReader(URL, ARCHIVE_SIZE)
    records = []
    with zipfile.ZipFile(reader) as archive:
        for run_id in RUNS:
            run_dir = OUT / str(run_id)
            run_dir.mkdir(exist_ok=True)
            for member in MEMBERS:
                name = f"{BASE}/run_id={run_id}/{member}"
                info = archive.getinfo(name)
                path = run_dir / Path(member).name
                fetch_stored(info, reader, path)
                records.append({"run_id": run_id, "member": name,
                                "relative_file": str(path.relative_to(ROOT)),
                                "size": info.file_size, "crc32": f"{info.CRC:08x}",
                                "sha256": sha256(path), "zip_crc_verified": True})
                print(f"run={run_id} {path.name} verified {info.file_size} bytes", flush=True)
    report = {"source": "https://zenodo.org/records/22260924",
              "source_license": "CC-BY-4.0", "archive": "DNg100_Stim_fullManc.zip",
              "archive_bytes": ARCHIVE_SIZE, "whole_archive_downloaded": False,
              "figure_code": "https://github.com/smpuglie/Pugliese_2026/blob/main/notebooks/Figure%204.ipynb",
              "figure_run_order": [30662613, 33195857, 33195876, 33195882],
              "previously_downloaded_run": 33195882, "new_runs": list(RUNS), "files": records}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"new_runs": len(RUNS), "new_files": len(records),
                      "bytes": sum(x["size"] for x in records)}, indent=2))


if __name__ == "__main__":
    main()
