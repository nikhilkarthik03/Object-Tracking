# Roadmap

Two parallel tracks. Track 1 incrementally improves the current SIFT+COLMAP+FAISS+PnP pipeline. Track 2 is a clean-room rebuild closer to OnePose++, optimized for edge deployment.

---

## Track 1 — Improve current pipeline

### 1.1 Clean the point cloud — object only, discard scene

**Problem:** COLMAP reconstructs everything in frame — background walls, tables, hands. The map centroid drifts, the gimbal lands in the wrong place, and background descriptors pollute the FAISS index causing false matches.

**Approaches (pick one or combine):**

**PCA bounding box filter**
The object occupies a compact cluster in 3D space. Background points are outliers along the dominant axes.
```python
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope

pca = PCA(n_components=3)
pca.fit(xyzs)
scores = pca.transform(xyzs)           # project to principal axes
detector = EllipticEnvelope(contamination=0.2)
mask = detector.fit_predict(scores)    # -1 = outlier
xyzs_clean = xyzs[mask == 1]
descs_clean = descs[mask == 1]
```

**Statistical outlier removal (like CloudCompare / Open3D)**
For each point, compute mean distance to its k nearest neighbours. Remove points whose mean distance exceeds mean + N*std.
```python
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyzs)
pcd_clean, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
xyzs_clean = xyzs[idx]
descs_clean = descs[idx]
```

**Reprojection-count filter**
Points that appear in many training images are almost certainly on the object surface, not the background (which moves in and out of frame). Already partly handled by `--min_views` but can be pushed higher (5–8) for object-only maps.

**Where to plug this in:** end of `build_map.py`, after `aggregate_descriptors()`, before `np.savez_compressed()`.

---

### 1.2 Use SfM poses to improve localization

**Problem:** We throw away the camera poses that COLMAP estimated for every training image. These are free ground-truth data that could dramatically improve localization.

**What COLMAP gives us:**
Each training image has a pose `(R_i, t_i)` stored in `images.bin`. This tells us exactly where the camera was when it took that training image.

**How to use them:**

**Pose-weighted centroid**
Instead of `xyzs.mean()` as the gimbal origin, use the centroid of points weighted by how many registered training views observe them — more reliable than a raw mean.

**Nearest-training-view initialization**
Before running FAISS search over all N map points, first retrieve the most similar training image using global image descriptors (NetVLAD, DINOv2). Then only search map points visible from that training camera's frustum. This is the core idea behind HLoc and cuts FAISS search space by ~10×.

```python
# pseudocode
def covisible_points(query_global_desc, training_global_descs, training_poses, xyzs, K=5):
    # find K nearest training images
    scores = training_global_descs @ query_global_desc
    top_k = np.argsort(scores)[-K:]
    # collect 3D points visible in those K cameras
    visible_mask = np.zeros(len(xyzs), dtype=bool)
    for i in top_k:
        R, t = training_poses[i]
        # project all points into camera i, keep those inside FOV
        pts_cam = (R @ xyzs.T).T + t
        visible_mask |= pts_cam[:, 2] > 0   # crude: in front of camera
    return visible_mask
```

**Read poses from COLMAP `images.bin`** — add a `read_images_binary()` function to `build_map.py` (same struct pattern as `read_cameras_binary` in `localize.py`) and save `poses` alongside `xyzs` and `descs` in `map.npz`.

---

### 1.3 Refer to OnePose++ to improve matching

**Problem:** SIFT fails on low-textured objects (bottles, cups, plastic containers). The ratio test drops too many matches and PnP gets fewer than 6 inliers.

**OnePose++ key ideas applicable here:**

| OnePose++ idea | How to apply to current code |
|---|---|
| LoFTR detector-free matching | Replace `extract_sift()` in `localize.py` with LoFTR query→database-image matching, then lift 2D–2D matches to 2D–3D using the SfM point track |
| Semi-dense point cloud | Increase `--max_descs_per_point` and lower `--min_views` to keep more points per surface patch |
| Coarse-to-fine 2D–3D matching | First match at 1/8 resolution, then refine to sub-pixel. Currently we do full-resolution SIFT → one-shot FAISS with no refinement |
| Mutual nearest neighbour (MNN) filter | Already implemented in `match_2d3d()` — but currently only uses ratio test. Add MNN: match query→db AND db→query, keep only symmetric matches |

**Immediate win — add MNN to `match_2d3d()`:**
```python
def match_2d3d_mnn(query_desc, db_desc, index_qd, index_dq, ratio=0.85):
    # forward: query → db
    D_fwd, I_fwd = index_qd.search(query_desc, 2)
    # backward: db → query  (need a second index built on query_desc)
    D_bwd, I_bwd = index_dq.search(db_desc[I_fwd[:, 0]], 2)
    good = []
    for i in range(len(query_desc)):
        ratio_ok = D_fwd[i, 0] < ratio * D_fwd[i, 1]
        mutual = I_bwd[i, 0] == i          # db's best match points back to query point i
        if ratio_ok and mutual:
            good.append((i, I_fwd[i, 0]))
    return good
```

---

### 1.4 GPU pipeline end-to-end

**Current bottlenecks:**

| Step | Current | GPU option |
|---|---|---|
| SIFT extraction | OpenCV CPU | `cuSIFT` or `kornia.feature.SIFT` on CUDA |
| FAISS search | CPU index | `faiss.index_cpu_to_gpu()` — already stubbed in `localize.py` with `--use_gpu` |
| PnP RANSAC | OpenCV CPU | No standard GPU PnP — use `poselib` which is faster CPU |
| Video decode | `cv2.imread` per frame | `cv2.VideoCapture` + `cv2.cuda.GpuMat` |
| ffmpeg encode | CPU | `ffmpeg -c:v h264_nvenc` (NVENC) |

**Steps to enable GPU pipeline:**
1. Enable `--use_gpu` in `tracker.sh` — this alone moves FAISS to GPU, which is the biggest search bottleneck
2. Install `kornia`: `conda install -c conda-forge kornia` — drop-in replacement for SIFT with CUDA support
3. Replace ffmpeg `libx264` with `h264_nvenc` in `tracker.sh` for GPU-accelerated encoding (also fixes the codec missing issue)

```bash
# GPU-accelerated ffmpeg encode
ffmpeg -y \
    -framerate 30 \
    -i "$FRAMES_DIR/frame_%06d.jpg" \
    -c:v h264_nvenc \
    -preset p4 \
    -cq 20 \
    "$VIDEO_OUT"
```

---

### 1.5 Edge device deployment

**Target constraints:** Jetson Orin / Raspberry Pi 5 — limited RAM, no discrete GPU, need <100ms per frame.

**Optimizations:**

**Quantize the FAISS index**
```python
# Replace IndexFlatL2 with a quantized index — 4x smaller, ~3x faster search
d = descs.shape[1]
nlist = 64                          # number of Voronoi cells
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(descs)
index.add(descs)
index.nprobe = 8                    # search 8 cells per query
```

**Reduce descriptor dimensionality with PCA**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=64)          # 128 → 64 dims, 2x speedup in FAISS
descs_reduced = pca.fit_transform(descs)
# save pca model alongside map.npz and apply same transform at query time
```

**Resize query images before SIFT**
Most of the SIFT time is spent on large images. Resize to 640px longest side before extraction — barely affects match quality on close-up object images.

**ONNX export for any learned features**
If switching to SuperPoint or similar: export to ONNX → run with TensorRT on Jetson or CoreML on Apple Silicon.

---

## Track 2 — OnePose++-style model

Build a learned end-to-end system following the OnePose++ architecture. No SIFT, no hand-crafted descriptors.

### 2.1 Architecture overview

```
Reference images + poses
        ↓
LoFTR detector-free matching between all image pairs
        ↓
Keypoint-free SfM (coarse COLMAP triangulation → sub-pixel refinement)
        ↓
Semi-dense point cloud {P_j} with LoFTR fine features stored per point
        ↓
                    ┌────────────────────────────────────┐
Query image ──────► │  2D-3D Matching Network            │
                    │  - ResNet-18 backbone               │
                    │  - Self + Cross attention (Nc=3)    │
                    │  - Coarse match → fine sub-pixel    │
                    └──────────────┬─────────────────────┘
                                   ↓
                               PnP + RANSAC
                                   ↓
                                 Pose
```

### 2.2 Implementation plan

**Phase 1 — Keypoint-free SfM**
- Replace `offline.sh` SIFT extraction with LoFTR pairwise matching
- Feed LoFTR coarse matches into COLMAP triangulation (coarse `points3D`)
- Refinement: for each coarse 3D point, fix one reference view, run LoFTR fine matching to all other views in the track, optimize depth by minimizing reprojection error (equation 1 in paper)
- Store per-point LoFTR fine features in `map.npz` instead of SIFT descriptors

**Phase 2 — 2D-3D matching network**
- Image backbone: ResNet-18 pretrained, extract 1/8 and 1/2 resolution feature maps
- Coarse module: flatten 2D feature map, apply Nc=3 self+cross attention layers with 3D point features, dual-softmax score matrix, MNN threshold θ=0.4
- Fine module: for each coarse match, crop w×w=5×5 window from 1/2-res feature map, Nf=1 self+cross attention, expectation over softmax scores → sub-pixel 2D location
- Loss: focal loss on coarse matches + L2 loss on fine 2D coordinates

**Phase 3 — Training**
- Train on OnePose dataset (450 sequences, 150 objects) — publicly available
- Random sample / pad point cloud to 7000 points per training example
- AdamW lr=4e-3, batch=32, ~20h on 8× V100 (or longer on smaller setup)
- Initialize backbone from LoFTR outdoor checkpoint

**Phase 4 — Optimization and size reduction**
- Quantize attention layers to INT8 with PyTorch `torch.quantization`
- Prune ResNet-18 backbone (remove last 2 blocks — not needed for local features)
- Distill into a smaller student (MobileNetV3 backbone + 1 attention layer)
- Target: <10MB model, <50ms per frame on Jetson Orin NX

**Phase 5 — Edge testing**
- Export to ONNX → TensorRT engine on Jetson
- Benchmark latency vs accuracy tradeoff on OnePose-LowTexture dataset
- Compare against Track 1 SIFT baseline on same hardware

### 2.3 Immediate next step to start Track 2

Use the official OnePose++ repo as a reference implementation while building our own:
- `https://github.com/zju3dv/OnePose_Plus_Plus`
- Start by running their inference on our bottle dataset to establish a quality ceiling
- Then reimplement piece by piece, starting with the SfM stage since we already have the COLMAP scaffolding

---

## Priority order

```
Track 1.1  Object-only point cloud      ← do first, fixes gimbal + match quality immediately
Track 1.2  Use SfM poses               ← do second, free data we're ignoring
Track 1.3  MNN matching                ← quick code change, meaningful accuracy gain
Track 1.4  GPU pipeline                ← enable --use_gpu flag first (already coded)
Track 2.1  Keypoint-free SfM           ← start in parallel with 1.3
Track 2.2  2D-3D matching network      ← after 2.1 produces a clean point cloud
Track 1.5 / 2.4  Edge deployment       ← last, after accuracy is satisfactory
```