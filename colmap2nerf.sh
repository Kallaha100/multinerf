#!/usr/bin/env bash
set -euo pipefail

# Defaults (as you requested)
IMAGES_DIR="${IMAGES_DIR:-/workspace/images}"
OUT_ROOT="${OUT_ROOT:-/workspace/data}"
SPARSE_DIR="$OUT_ROOT/sparse"
DB_PATH="$OUT_ROOT/database.db"
UNDISTORT="${UNDISTORT:-0}"        # 1 to enable undistortion step
SINGLE_CAMERA="${SINGLE_CAMERA:-1}"# 1 for phone shots / single intrinsics
SCENE_SCALE="${SCENE_SCALE:-4.0}"  # rescale so median cam distance ~= this
N_THREADS="${N_THREADS:-$(nproc)}" # for matching & mapping

mkdir -p "$OUT_ROOT" "$SPARSE_DIR"

# --- sanity checks ---
if [ ! -d "$IMAGES_DIR" ]; then
  echo "ERROR: IMAGES_DIR not found: $IMAGES_DIR" >&2
  exit 1
fi

# Count images (jpg/png/jpeg)
shopt -s nullglob nocaseglob
mapfile -t IMGS < <(printf "%s\0" "$IMAGES_DIR"/*.{jpg,jpeg,png} | xargs -0 -n1 echo || true)
NUM_IMGS=${#IMGS[@]}
if (( NUM_IMGS == 0 )); then
  echo "ERROR: No images found in $IMAGES_DIR (jpg/jpeg/png)" >&2
  exit 1
fi
echo "Found $NUM_IMGS images."

# --- choose matcher automatically ---
# heuristic: <= 600 ⇒ exhaustive; else sequential (good for video/walkaround)
MATCHER="${MATCHER:-auto}"
if [ "$MATCHER" = "auto" ]; then
  if (( NUM_IMGS <= 600 )); then
    MATCHER="exhaustive"
  else
    MATCHER="sequential"
  fi
fi
echo "Using matcher: $MATCHER"

# fresh database for each run
rm -f "$DB_PATH"

# 1) Feature extraction
colmap feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$IMAGES_DIR" \
  --ImageReader.single_camera $SINGLE_CAMERA \
  --SiftExtraction.use_gpu 1 \
  --SiftExtraction.num_threads $N_THREADS

# 2) Matching
case "$MATCHER" in
  exhaustive)
    colmap exhaustive_matcher \
      --database_path "$DB_PATH" \
      --SiftMatching.use_gpu 1 \
      --SiftMatching.num_threads $N_THREADS
    ;;
  sequential)
    colmap sequential_matcher \
      --database_path "$DB_PATH" \
      --SiftMatching.use_gpu 1 \
      --SiftMatching.num_threads $N_THREADS
    ;;
  *)
    echo "ERROR: unknown MATCHER='$MATCHER' (use exhaustive|sequential|auto)" >&2
    exit 1
    ;;
esac

# 3) Sparse mapping
colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$IMAGES_DIR" \
  --output_path "$SPARSE_DIR" \
  --Mapper.num_threads $N_THREADS

# pick the first reconstruction (sparse/0)
REC_DIR="$SPARSE_DIR/0"
if [ ! -d "$REC_DIR" ]; then
  echo "ERROR: no reconstruction found in $SPARSE_DIR" >&2
  exit 1
fi

# Optional undistortion
if [ "$UNDISTORT" = "1" ]; then
  UNDIST_DIR="$OUT_ROOT/undistorted"
  mkdir -p "$UNDIST_DIR"
  colmap image_undistorter \
    --image_path "$IMAGES_DIR" \
    --input_path "$REC_DIR" \
    --output_path "$UNDIST_DIR" \
    --output_type COLMAP
  IMG_PATH_FOR_JSON="$UNDIST_DIR/images"
else
  IMG_PATH_FOR_JSON="$IMAGES_DIR"
fi

# 4) Write transforms.json (Blender-style) via a tiny converter
# Install converter if missing
CONVERTER="/usr/local/bin/colmap2nerf.py"
if [ ! -f "$CONVERTER" ]; then
  mkdir -p /usr/local/bin
  cat > "$CONVERTER" << 'PY'
#!/usr/bin/env python3
import argparse, json, os, numpy as np
import pycolmap

def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z,   2*x*y-2*z*w,     2*x*z+2*y*w],
        [2*x*y+2*z*w,     1-2*x*x-2*z*z,   2*y*z-2*x*w],
        [2*x*z-2*y*w,     2*y*z+2*x*w,     1-2*x*x-2*y*y],
    ], dtype=np.float64)

def colmap_cam_to_nerf_blender(c2w):
    # Convert COLMAP (cam looks +Z, y down) -> Blender/NeRF (cam looks -Z, y up)
    fix = np.diag([1, -1, -1, 1])
    return c2w @ fix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap_sparse", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="transforms.json")
    ap.add_argument("--scene_scale", type=float, default=None)
    args = ap.parse_args()

    rec = pycolmap.Reconstruction(args.colmap_sparse)
    if len(rec.images) == 0:
        raise RuntimeError("No images found in reconstruction.")

    # assume single shared intrinsics (common with single_camera=1)
    some_img = next(iter(rec.images.values()))
    cam = rec.cameras[some_img.camera_id]
    w, h = cam.width, cam.height

    # intrinsics
    if cam.model in ("PINHOLE", "SIMPLE_PINHOLE"):
        fx = cam.params[0]
        fy = cam.params[1] if cam.model == "PINHOLE" else fx
        cx = cam.params[-2]
        cy = cam.params[-1]
    elif cam.model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "SIMPLE_RADIAL", "RADIAL", "RADIAL_FISHEYE"):
        fx = cam.params[0]; fy = cam.params[1]; cx = cam.params[2]; cy = cam.params[3]
    else:
        raise NotImplementedError(f"Unsupported camera model: {cam.model}")

    frames, c2w_list = [], []
    for img in rec.images.values():
        Rwc = qvec2rotmat(img.qvec).T
        twc = -Rwc @ img.tvec
        c2w = np.eye(4)
        c2w[:3,:3] = Rwc
        c2w[:3, 3] = twc
        c2w = colmap_cam_to_nerf_blender(c2w)
        c2w_list.append(c2w)

        # relative path if possible
        name = img.name
        try:
            rel = os.path.relpath(name, start=args.images)
            if rel.startswith(".."):
                rel = os.path.basename(name)
        except Exception:
            rel = os.path.basename(name)

        frames.append({
            "file_path": os.path.join(os.path.basename(args.images), rel).replace("\\", "/"),
            "transform_matrix": c2w.tolist(),
        })

    if args.scene_scale is not None:
        dists = [np.linalg.norm(M[:3,3]) for M in c2w_list]
        med = float(np.median(dists))
        if med > 1e-6:
            s = args.scene_scale / med
            for f in frames:
                M = np.array(f["transform_matrix"])
                M[:3,3] *= s
                f["transform_matrix"] = M.tolist()

    out = {
        "fl_x": float(fx), "fl_y": float(fy),
        "cx": float(cx), "cy": float(cy),
        "w": int(w), "h": int(h),
        "camera_model": cam.model,
        "frames": frames,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out} with {len(frames)} frames")

if __name__ == "__main__":
    main()
PY
  chmod +x "$CONVERTER"
fi

TRANS_JSON="$OUT_ROOT/transforms.json"
python "$CONVERTER" \
  --colmap_sparse "$REC_DIR" \
  --images "$IMG_PATH_FOR_JSON" \
  --out "$TRANS_JSON" \
  --scene_scale "$SCENE_SCALE"

echo "Done."
echo "COLMAP DB:     $DB_PATH"
echo "Sparse model:  $REC_DIR"
[ "$UNDISTORT" = "1" ] && echo "Undistorted:   $IMG_PATH_FOR_JSON"
echo "Transforms:    $TRANS_JSON"
