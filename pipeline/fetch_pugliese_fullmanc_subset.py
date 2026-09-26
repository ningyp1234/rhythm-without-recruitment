"""Range-fetch the official full-MANC parameters without its 232 MB trace."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import zipfile

from fetch_pugliese_zenodo_subset import HTTPRangeReader


URL = "https://zenodo.org/api/records/22260924/files/DNg100_Stim_fullManc.zip/content"
ARCHIVE_SIZE = 2_030_250_053
RUN = 33_195_882
PREFIX = f"simulations/DNg100_Stim_fullManc/hyak/run_id={RUN}/"
MEMBERS = [
    PREFIX + ".hydra/config.yaml",
    PREFIX + ".hydra/overrides.yaml",
    PREFIX + "logs/run_config.yaml",
    PREFIX + "ckpt/neuron_params.h5",
]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/pugliese-zenodo-22260924/full-manc-run33195882"


def fetch_stored(info, reader, destination):
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"Expected stored member: {info.filename}")
    reader.seek(info.header_offset)
    fields = struct.unpack(zipfile.structFileHeader, reader.read(zipfile.sizeFileHeader))
    data_offset = (
        info.header_offset
        + zipfile.sizeFileHeader
        + fields[zipfile._FH_FILENAME_LENGTH]
        + fields[zipfile._FH_EXTRA_FIELD_LENGTH]
    )
    done = destination.stat().st_size if destination.exists() else 0
    if done > info.file_size:
        destination.write_bytes(b"")
        done = 0
    while done < info.file_size:
        request = urllib.request.Request(
            URL,
            headers={"Range": f"bytes={data_offset + done}-{data_offset + info.file_size - 1}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise IOError(f"Range request returned {response.status}")
                with destination.open("ab") as target:
                    while done < info.file_size:
                        chunk = response.read(min(1024 * 1024, info.file_size - done))
                        if not chunk:
                            raise IOError("Remote member ended early")
                        target.write(chunk)
                        done += len(chunk)
                        if done % (10 * 1024 * 1024) < len(chunk):
                            print(destination.name, done, flush=True)
        except Exception as exc:
            print(destination.name, "resume", done, repr(exc), flush=True)
            time.sleep(2)
    if (zipfile.crc32(destination.read_bytes()) & 0xFFFFFFFF) != info.CRC:
        raise IOError(f"CRC mismatch: {destination}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reader = HTTPRangeReader(URL, ARCHIVE_SIZE)
    files = []
    with zipfile.ZipFile(reader) as archive:
        for name in MEMBERS:
            info = archive.getinfo(name)
            destination = OUT / Path(name).name
            fetch_stored(info, reader, destination)
            files.append(
                {
                    "member": name,
                    "file": str(destination),
                    "size": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                }
            )
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "record": "https://zenodo.org/records/22260924",
                "doi": "10.5281/zenodo.22260924",
                "record_created": "2026-09-15",
                "archive": "DNg100_Stim_fullManc.zip",
                "archive_md5": "0f3d47ef1f6d33421e6f53ee25e74120",
                "archive_size": ARCHIVE_SIZE,
                "license": "CC-BY-4.0",
                "selection": (
                    "official run 33195882, seed 424, bilateral DNg100 current 380; "
                    "the trace archive is deliberately omitted because local dynamics are rerun"
                ),
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"downloaded": len(files), "bytes": sum(x["size"] for x in files)}))


if __name__ == "__main__":
    main()
