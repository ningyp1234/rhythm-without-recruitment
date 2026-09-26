"""Fetch only the official CoreCPG members needed for local reproduction.

The Zenodo archive is 487 MB and stores every member without compression.
HTTP range requests retrieve the central directory and selected members while
zipfile checks CRC32; the entire archive need not be downloaded.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct
import time
import urllib.request
import zipfile


URL = "https://zenodo.org/api/records/22260924/files/DNg100_Stim_CoreCPG.zip/content"
ARCHIVE_SIZE = 487_438_909
PREFIX = (
    "simulations/DNg100_Stim_CoreCPG/hyak/run_id=29278022/"
    "experiment.seed=207,experiment.stimI=[300]/"
)
MEMBERS = [
    PREFIX + ".hydra/config.yaml",
    PREFIX + ".hydra/overrides.yaml",
    PREFIX + "logs/run_config.yaml",
    PREFIX + "ckpt/neuron_params.h5",
    PREFIX + "ckpt/DNg100_Stim_CoreCPG_Rs.npz",
]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/pugliese-zenodo-22260924/core-cpg-seed207-stim300"


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url, size):
        self.url = url
        self.size = size
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {
            io.SEEK_SET: offset,
            io.SEEK_CUR: self.pos + offset,
            io.SEEK_END: self.size + offset,
        }[whence]
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.pos
        if size <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size - 1, self.pos + size - 1)
        expected = end - self.pos + 1
        error = None
        for attempt in range(6):
            request = urllib.request.Request(
                self.url, headers={"Range": f"bytes={self.pos}-{end}"}
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                    if response.status != 206:
                        raise IOError(f"Range request returned {response.status}")
                if len(data) != expected:
                    raise IOError(f"Range returned {len(data)} of {expected} bytes")
                break
            except Exception as exc:
                error = exc
                if attempt == 5:
                    raise IOError(f"Range request failed after retries: {error}") from error
                time.sleep(2 ** attempt)
        self.pos += len(data)
        return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reader = HTTPRangeReader(URL, ARCHIVE_SIZE)
    manifest = []
    with zipfile.ZipFile(reader) as archive:
        for name in MEMBERS:
            info = archive.getinfo(name)
            destination = OUT / Path(name).name
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"Selective fetch requires stored member: {name}")
            reader.seek(info.header_offset)
            local_header = reader.read(zipfile.sizeFileHeader)
            fields = struct.unpack(zipfile.structFileHeader, local_header)
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
                    headers={
                        "Range": (
                            f"bytes={data_offset + done}-"
                            f"{data_offset + info.file_size - 1}"
                        )
                    },
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
                raise IOError(f"CRC mismatch after selective fetch: {destination}")
            manifest.append(
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
                "archive": "DNg100_Stim_CoreCPG.zip",
                "archive_md5": "48bf37609e2ce4728e869c845fafe18b",
                "archive_size": ARCHIVE_SIZE,
                "license": "CC-BY-4.0",
                "selection": (
                    "official run 29278022, experiment seed 207, stimulus 300; "
                    "selected because official scores.csv reports parameter row 207 score 1.0"
                ),
                "files": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "downloaded": len(manifest),
                "bytes": sum(item["size"] for item in manifest),
                "out": str(OUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
