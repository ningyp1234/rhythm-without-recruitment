"""Range-fetch the exact published full-MANC run33195882 neural trajectory.

The 2.03 GB archive itself is not downloaded. The stored 200 MB NPZ member is
retrieved with resumable HTTP ranges and checked against the ZIP CRC32.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from fetch_pugliese_fullmanc_subset import (
    ARCHIVE_SIZE, OUT, PREFIX, URL, fetch_stored,
)
from fetch_pugliese_zenodo_subset import HTTPRangeReader


MEMBER = PREFIX + "ckpt/DNg100_Stim_fullManc_Rs.npz"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reader = HTTPRangeReader(URL, ARCHIVE_SIZE)
    with zipfile.ZipFile(reader) as archive:
        info = archive.getinfo(MEMBER)
        destination = OUT / Path(MEMBER).name
        fetch_stored(info, reader, destination)
    data = destination.read_bytes()
    item = {"member": MEMBER, "file": str(destination), "size": len(data),
            "crc32": f"{info.CRC:08x}", "sha256": hashlib.sha256(data).hexdigest(),
            "source": "https://zenodo.org/records/22260924",
            "archive": "DNg100_Stim_fullManc.zip", "license": "CC-BY-4.0",
            "zip_directory_member_size_and_crc_verified": len(data) == info.file_size and zipfile.crc32(data) == info.CRC,
            "whole_archive_downloaded": False}
    (OUT / "author_trace_manifest.json").write_text(json.dumps(item, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: item[k] for k in ("size", "crc32", "sha256", "zip_directory_member_size_and_crc_verified")}, indent=2))
    if not item["zip_directory_member_size_and_crc_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
