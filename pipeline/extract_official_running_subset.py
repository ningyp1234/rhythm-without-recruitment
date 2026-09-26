"""Build a small, provenance-preserving subset of the 2026 running dataset.

The input is the unmodified public ``Full_running_dataset.h5``.  Selection is
deterministic, grouped by source fly, and stratified by measured root speed.
No synthetic pose, interpolated frame, or target gait is generated here.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/official-3d-kinematics-20260916"
RAW = DATA / "raw/Full_running_dataset.h5"
SUBSET = DATA / "running-reference-subset.h5"
SUMMARY = DATA / "running-bout-index.csv"
MANIFEST = DATA / "manifest.json"
EXPECTED_BYTES = 3_303_870_746
FPS = 800.0
MODEL_TO_MM = 10.0
LEGS = ("T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right")
CLAW_NAMES = tuple(f"claw_{leg}" for leg in LEGS)
SOURCE_URL = (
    "https://drive.google.com/drive/folders/"
    "1flBiyFmJYWPA6EIN4Xoh_2Lc5VT2_AoV?usp=sharing"
)
SOURCE_FILE_ID = "1MDcnUChFsyGGRHY16ZMXALyUdYiVUGKF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_sequence(node) -> list:
    """Load a dataset or the integer-keyed groups written by io_dict_to_hdf5."""
    if isinstance(node, h5py.Dataset):
        value = node[()]
        if np.ndim(value) == 0:
            return [scalar(value)]
        return [scalar(v) for v in value]
    keys = sorted(node.keys(), key=lambda key: int(key))
    return [scalar(node[key][()]) for key in keys]


def copy_filtered(node, destination, selected: list[int], total_bouts: int):
    """Copy an HDF5 subtree, filtering only clearly per-bout first axes."""
    for key, child in node.items():
        if isinstance(child, h5py.Dataset):
            value = child[()]
            if np.ndim(value) > 0 and value.shape[0] == total_bouts:
                value = value[selected]
            kwargs = {}
            if np.ndim(value) > 0 and np.size(value) > 1:
                kwargs = {"compression": "gzip", "compression_opts": 4}
            out = destination.create_dataset(key, data=value, **kwargs)
            for attr, attr_value in child.attrs.items():
                out.attrs[attr] = attr_value
        else:
            keys = list(child.keys())
            integer_list = bool(keys) and all(k.isdigit() for k in keys)
            if integer_list and len(keys) == total_bouts:
                out = destination.create_group(key)
                for new_index, old_index in enumerate(selected):
                    child.copy(str(old_index), out, name=str(new_index))
            else:
                out = destination.create_group(key)
                copy_filtered(child, out, selected, total_bouts)
            for attr, attr_value in child.attrs.items():
                out.attrs[attr] = attr_value


def stable_split(fly_ids: list[str]) -> dict[str, str]:
    """Assign whole flies to 70/15/15 partitions with a stable hash."""
    unique = sorted(set(fly_ids))
    ranked = sorted(
        unique,
        key=lambda fly: hashlib.sha256(f"fly3d-v2|{fly}".encode()).digest(),
    )
    n = len(ranked)
    train_end = round(0.70 * n)
    val_end = train_end + round(0.15 * n)
    return {
        fly: ("train" if i < train_end else "validation" if i < val_end else "test")
        for i, fly in enumerate(ranked)
    }


def select_stratified(rows: list[dict], split: str, count: int) -> list[int]:
    candidates = [row for row in rows if row["split"] == split]
    targets = np.linspace(0.08, 0.92, count)
    speeds = np.asarray([row["mean_path_speed_mm_s"] for row in candidates])
    target_values = np.quantile(speeds, targets)
    picked: list[int] = []
    used_flies: set[str] = set()
    for target in target_values:
        choices = sorted(
            candidates,
            key=lambda row: (
                row["fly_id"] in used_flies,
                abs(row["mean_path_speed_mm_s"] - target),
                row["source_index"],
            ),
        )
        choice = next(row for row in choices if row["source_index"] not in picked)
        picked.append(choice["source_index"])
        used_flies.add(choice["fly_id"])
    return picked


def cycle_count(z_mm: np.ndarray) -> int:
    floor = float(np.percentile(z_mm, 5))
    height = z_mm - floor
    prominence = max(0.035, 0.15 * float(np.ptp(height)))
    peaks, _ = find_peaks(height, prominence=prominence, distance=round(0.020 * FPS))
    return int(len(peaks))


def main():
    if not RAW.exists() or RAW.stat().st_size != EXPECTED_BYTES:
        raise FileNotFoundError(
            f"Expected complete official file ({EXPECTED_BYTES} bytes), got "
            f"{RAW.stat().st_size if RAW.exists() else 'missing'}"
        )
    DATA.mkdir(parents=True, exist_ok=True)
    with h5py.File(RAW, "r") as source:
        bout_keys = sorted(key for key in source if key.startswith("bout_"))
        info = source["info"]
        clip_lengths = np.asarray(info["clip_lengths"][()]).astype(int)
        fly_ids = [str(value) for value in load_sequence(info["fly_ids"])]
        names_xpos = [str(value) for value in load_sequence(info["names_xpos"])]
        claw_indices = [names_xpos.index(name) for name in CLAW_NAMES]
        if len(bout_keys) != len(clip_lengths) or len(bout_keys) != len(fly_ids):
            raise ValueError("Official bout metadata lengths disagree")
        partitions = stable_split(fly_ids)
        rows: list[dict] = []
        for index, key in enumerate(bout_keys):
            group = source[key]
            qpos = np.asarray(group["qpos"])
            xpos = group["xpos"]
            root_xy = qpos[:, :2] * MODEL_TO_MM
            delta = np.diff(root_xy, axis=0)
            path_mm = float(np.linalg.norm(delta, axis=1).sum())
            net_mm = float(np.linalg.norm(root_xy[-1] - root_xy[0]))
            seconds = (len(qpos) - 1) / FPS
            cycles = [
                cycle_count(np.asarray(xpos[:, body_index, 2]) * MODEL_TO_MM)
                for body_index in claw_indices
            ]
            rows.append(
                {
                    "source_index": index,
                    "bout_key": key,
                    "fly_id": fly_ids[index],
                    "split": partitions[fly_ids[index]],
                    "frames": int(len(qpos)),
                    "seconds": seconds,
                    "path_mm": path_mm,
                    "net_displacement_mm": net_mm,
                    "mean_path_speed_mm_s": path_mm / seconds,
                    "cycles_T1L": cycles[0],
                    "cycles_T1R": cycles[1],
                    "cycles_T2L": cycles[2],
                    "cycles_T2R": cycles[3],
                    "cycles_T3L": cycles[4],
                    "cycles_T3R": cycles[5],
                }
            )
        selected = (
            select_stratified(rows, "train", 12)
            + select_stratified(rows, "validation", 3)
            + select_stratified(rows, "test", 3)
        )
        selected = sorted(selected)
        temporary = SUBSET.with_suffix(".h5.tmp")
        if temporary.exists():
            temporary.unlink()
        with h5py.File(temporary, "w") as destination:
            for attr, value in source.attrs.items():
                destination.attrs[attr] = value
            destination.attrs["subset_kind"] = "real_source_bouts_no_synthetic_frames"
            destination.attrs["source_url"] = SOURCE_URL
            destination.attrs["source_file_id"] = SOURCE_FILE_ID
            destination.attrs["source_expected_bytes"] = EXPECTED_BYTES
            destination.attrs["sampling_hz"] = FPS
            destination.attrs["selection"] = (
                "whole-fly 70/15/15 split; measured-speed stratification; 12/3/3 bouts"
            )
            info_out = destination.create_group("info")
            copy_filtered(info, info_out, selected, len(bout_keys))
            subset_info = destination.create_group("subset")
            string_type = h5py.string_dtype("utf-8")
            subset_info.create_dataset(
                "source_indices", data=np.asarray(selected, dtype=np.int32)
            )
            subset_info.create_dataset(
                "bout_keys",
                data=np.asarray([bout_keys[index] for index in selected], dtype=string_type),
            )
            subset_info.create_dataset(
                "splits",
                data=np.asarray([rows[index]["split"] for index in selected], dtype=string_type),
            )
            for index in selected:
                source.copy(bout_keys[index], destination)
        temporary.replace(SUBSET)

    fields = list(rows[0])
    with SUMMARY.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    raw_sha = sha256(RAW)
    subset_sha = sha256(SUBSET)
    chosen = [rows[index] for index in selected]
    manifest = {
        "source": {
            "paper": "https://www.biorxiv.org/content/10.64898/2026.05.03.722293v2",
            "data_folder": SOURCE_URL,
            "file_id": SOURCE_FILE_ID,
            "filename": RAW.name,
            "bytes": RAW.stat().st_size,
            "sha256": raw_sha,
            "article_license": "CC-BY-4.0",
            "data_specific_license": "No separate license file was present in the public folder; do not infer beyond the article license.",
        },
        "official_dataset_observed": {
            "running_bouts": len(rows),
            "source_flies": len(set(fly_ids)),
            "frames": int(sum(row["frames"] for row in rows)),
            "sampling_hz": FPS,
        },
        "split_policy": {
            "unit": "fly_id",
            "counts_by_fly": {
                split: sum(value == split for value in partitions.values())
                for split in ("train", "validation", "test")
            },
            "frame_or_bout_leakage_between_splits": False,
        },
        "subset": {
            "path": str(SUBSET.relative_to(ROOT)),
            "bytes": SUBSET.stat().st_size,
            "sha256": subset_sha,
            "bouts": len(chosen),
            "flies": len(set(row["fly_id"] for row in chosen)),
            "frames": sum(row["frames"] for row in chosen),
            "splits": {
                split: sum(row["split"] == split for row in chosen)
                for split in ("train", "validation", "test")
            },
            "selection": "measured-speed-stratified; whole flies remain in one partition",
            "synthetic_frames": 0,
        },
        "selected_bouts": chosen,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "full_bouts": len(rows),
                "full_flies": len(set(fly_ids)),
                "full_frames": manifest["official_dataset_observed"]["frames"],
                "subset_bouts": len(chosen),
                "subset_flies": manifest["subset"]["flies"],
                "subset_bytes": SUBSET.stat().st_size,
                "raw_sha256": raw_sha,
                "subset_sha256": subset_sha,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
