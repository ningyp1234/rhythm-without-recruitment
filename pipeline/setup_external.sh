#!/usr/bin/env bash
# Clone the external code used by the pipeline, at the commits used in the
# original runs, into external/work/ (ignored by git). Run from the repository root.
set -euo pipefail
mkdir -p external/work external/work/upstream-20260923

clone_at () {  # url dir commit
  if [ ! -d "$2/.git" ]; then git clone --quiet --filter=blob:none "$1" "$2"; fi
  git -C "$2" fetch --quiet origin "$3" || true
  git -C "$2" checkout --quiet "$3"
  echo "$2 -> $(git -C "$2" rev-parse HEAD)"
}

# FlyGym (Apache-2.0). source_calibration.py checks this exact commit.
clone_at https://github.com/NeLy-EPFL/flygym.git external/work/flygym 38c8ec61034cd59bc5ba0de20688d4a3c0000d60
python -m pip install -e external/work/flygym

# 3d_tracking_ik (MIT): provides model/fruitfly_v1/fruitfly_v1_free.xml
clone_at https://github.com/elliottabe/3d_tracking_ik.git external/work/upstream-20260923/3d_tracking_ik defdb66932a17e41c8852f3b4c5056bee74046eb
test -f external/work/upstream-20260923/3d_tracking_ik/model/fruitfly_v1/fruitfly_v1_free.xml && echo "fruitfly_v1 body found"
