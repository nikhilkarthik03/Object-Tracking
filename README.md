# Object Tracking

Visual localization pipeline that builds a 3D map of an object from a set of images, then estimates camera pose on any new image of that object and draws a gimbal (coordinate axes) at the object's center.

Built on COLMAP (structure-from-motion), SIFT descriptors, FAISS nearest-neighbor search, and OpenCV PnP RANSAC.

---

## How it works

```
OFFLINE  ──  images → COLMAP → 3D point cloud + descriptors → map.npz

ONLINE   ──  query image → SIFT → match against map → PnP RANSAC → pose (R, t) → gimbal overlay
```

**Stage 1 — COLMAP reconstruction (`offline.sh`)**
Runs SIFT feature extraction, exhaustive matching, and sparse SfM reconstruction on your training images. Produces `sparse/0/points3D.bin`, `cameras.bin`, and `database.db`.

**Stage 2 — Map build (`build_map.py`)**
Reads COLMAP output and builds a searchable FAISS descriptor index. Each 3D point gets up to 4 maximally-diverse SIFT descriptors via greedy farthest-point sampling — this preserves viewpoint variation and improves match recall on unseen query images.

**Stage 3 — Localization (`localize.py`)**
Extracts SIFT from a query image, searches the FAISS index for 2D–3D correspondences, runs PnP RANSAC to recover camera pose `(R, t)`, and projects a gimbal at the object centroid.

**Stage 4 — Video (`tracker.sh`)**
Runs localization over an entire folder of images, writes annotated frames, and stitches them into `tracking_result.mp4` via ffmpeg. Frames where localization fails are included as-is with a small `no pose` label so the video stays continuous and you can judge accuracy across the sequence.

---

## File structure

```
object-tracking/
├── offline.sh          # Stage 1 — run COLMAP reconstruction
├── build_map.py        # Stage 2 — build FAISS descriptor map from COLMAP output
├── localize.py         # Stage 3 — localize a single query image
├── tracker.sh          # Stage 4 — batch localize all images + render video
│
└── bottle/             # example scene  (one object = one folder)
    ├── images/         # training images used to build the map
    │   ├── image_0001.jpg
    │   └── ...
    ├── database.db     # COLMAP feature database       (auto-generated)
    ├── sparse/
    │   └── 0/
    │       ├── cameras.bin     # camera intrinsics
    │       ├── images.bin      # registered image poses
    │       └── points3D.bin    # 3D point cloud
    ├── map.npz          # descriptor index              (auto-generated)
    ├── frames/          # per-frame annotated images    (auto-generated)
    └── tracking_result.mp4
```

---

## Requirements

- [COLMAP](https://colmap.github.io/) — must be on `$PATH`
- Python 3.10+ conda environment:

```bash
conda create -n object-tracking python=3.10
conda activate object-tracking
conda install -c conda-forge faiss-cpu opencv numpy ffmpeg
```

> **pyenv users:** pyenv shims intercept `python3` before conda gets a chance.
> Run `pyenv local system` inside the project directory once, then reactivate
> your conda env. Alternatively, update the `PYTHON=` variable at the top of
> `tracker.sh` to the full conda Python path (e.g.
> `/home/user/miniconda3/envs/object-tracking/bin/python3`).

---

## Quick start

### 1. Prepare training images

Create a scene folder and put 50–200 images of your object inside `images/`. Images should cover the object from many angles with good overlap between adjacent views.

```bash
mkdir -p ./bottle/images
cp /path/to/your/images/*.jpg ./bottle/images/
```

### 2. Run the full pipeline

```bash
./tracker.sh ./bottle ./bottle/images
```

This will automatically:
1. Run COLMAP if `sparse/0/points3D.bin` does not exist yet
2. Build `map.npz` if it does not exist yet
3. Localize every image in the query folder
4. Write annotated frames to `bottle/frames/`
5. Render `bottle/tracking_result.mp4`

### 3. Localize a single image

```bash
python3 localize.py \
    --map_path     ./bottle/map.npz \
    --image_path   ./query.jpg \
    --cameras_path ./bottle/sparse/0/cameras.bin \
    --output_path  ./result.jpg \
    --gimbal_scale 0.3
```

### 4. Rebuild everything from scratch

```bash
rm -rf ./bottle/sparse ./bottle/database.db ./bottle/map.npz ./bottle/frames
./tracker.sh ./bottle ./bottle/images
```

---

## Parameters

### `build_map.py`

| Flag | Default | Description |
|---|---|---|
| `--min_views` | `3` | Minimum times a 3D point must be observed to be kept. Raise to filter noisy points. |
| `--max_descs_per_point` | `4` | Max SIFT descriptors stored per 3D point. Higher = better recall, larger index. |
| `--no_normalize` | off | Disable L2 normalization of descriptors before indexing. |

### `localize.py`

| Flag | Default | Description |
|---|---|---|
| `--ratio` | `0.75` | Lowe ratio test threshold. Lower = stricter, fewer but more correct matches. |
| `--ransac_thresh` | `5.0` | PnP reprojection error in pixels. Lower = stricter inlier requirement. |
| `--ransac_iters` | `1000` | RANSAC iterations. Raise if inlier rate is low. |
| `--gimbal_scale` | `0.3` | Length of gimbal arms in world units. Tune to match your reconstruction scale. |
| `--use_gpu` | off | Use GPU FAISS index (requires faiss-gpu). |

---

## Troubleshooting

**All frames show `no pose`**

Remove the `2>/dev/null` stderr suppression from `tracker.sh` temporarily and run one image manually to see the real error:

```bash
python3 localize.py \
    --map_path     ./bottle/map.npz \
    --image_path   ./bottle/images/image_0001.jpg \
    --cameras_path ./bottle/sparse/0/cameras.bin \
    --output_path  ./test.jpg \
    --ratio 0.85 \
    --ransac_thresh 12.0 \
    --ransac_iters 3000
```

**PnP RANSAC failed — not enough inliers**

Try relaxing the matching thresholds:

```bash
--ratio 0.85 --ransac_thresh 12.0 --ransac_iters 5000
```

If it still fails, check map quality (see below) and make sure your training images have good angular coverage of the object.

**`No module named faiss`**

pyenv is intercepting `python3`. Fix:

```bash
cd ~/your/project
pyenv local system
conda activate object-tracking
```

Or update the `PYTHON=` line at the top of `tracker.sh` to the full path of your conda Python.

**`Unknown encoder libx264`**

Your ffmpeg build excludes GPL codecs. Reinstall with codec support:

```bash
conda install -c conda-forge x264 x265 ffmpeg --force-reinstall
```

**Gimbal appears in the wrong place or off-screen**

The gimbal is drawn at the mean centroid of all map points. If COLMAP included background clutter in the reconstruction, the centroid drifts away from the object. Crop your training images tightly to the object before running COLMAP, or increase `--min_views` to filter out background points that appear in fewer frames.

**Check map quality**

```bash
python3 - <<'EOF'
import numpy as np
d = np.load("./bottle/map.npz")
xyzs, descs = d["xyzs"], d["descs"]
print(f"Descriptor entries : {len(xyzs)}")
print(f"XYZ range X        : {xyzs[:,0].min():.3f}  to  {xyzs[:,0].max():.3f}")
print(f"XYZ range Y        : {xyzs[:,1].min():.3f}  to  {xyzs[:,1].max():.3f}")
print(f"XYZ range Z        : {xyzs[:,2].min():.3f}  to  {xyzs[:,2].max():.3f}")
print(f"Centroid           : {xyzs.mean(axis=0)}")
EOF
```

A healthy map has at least 5000 descriptor entries. Fewer than that usually means COLMAP only registered a subset of your images — check the COLMAP output for registration warnings.

---

## Limitations

- Uses SIFT, so performance degrades on textureless or highly reflective objects
- No temporal tracking between frames — each frame is localized independently from scratch
- Gimbal origin is the mean centroid of all map points — works well for compact objects, may need adjustment for large or cluttered scenes
- Full FAISS search on every frame — not optimised for real-time use

## Potential next steps

- Frame-to-frame optical flow tracking to reuse matches and smooth the gimbal
- Learned features (SuperPoint + SuperGlue) for better performance on difficult objects
- Covisibility filtering to only search map points likely visible from the current viewpoint
- Online map updates as new images are added

## Video Generation

```bash
ffmpeg -y -framerate 10 -pattern_type glob -i "./bottle/frames/*.jpg" -c:v libopenh264 -b:v 2M -pix_fmt yuv420p ./bottle/result.mp4
```


