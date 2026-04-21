# Resources

Mathematical foundations behind every step of the pipeline. No hand-waving.

---

## COLMAP — Structure from Motion (SfM)

COLMAP implements incremental SfM. The goal is: given N unordered images, recover the 3D structure of the scene and the 6-DoF camera pose for every image simultaneously.

### Step 1 — Feature extraction

COLMAP uses SIFT (Scale-Invariant Feature Transform). For each image `I`, detect keypoints and compute 128-dimensional descriptors.

**Scale space extrema detection**

Build a Gaussian scale space by convolving the image with Gaussians at increasing σ:

```
L(x, y, σ) = G(x, y, σ) * I(x, y)

where G(x, y, σ) = (1 / 2πσ²) · exp(-(x² + y²) / 2σ²)
```

Difference of Gaussians (DoG) approximates the Laplacian of Gaussian:

```
D(x, y, σ) = L(x, y, kσ) - L(x, y, σ)
```

Keypoints are local extrema (maxima and minima) of D across both space (x, y) and scale σ. A candidate (x, y, σ) is a keypoint if D(x, y, σ) > all 26 neighbours in the 3×3×3 spatial-scale neighbourhood.

**Sub-pixel refinement**

Fit a 3D quadratic to the DoG values around the candidate using a Taylor expansion:

```
D(x) ≈ D + (∂D/∂x)ᵀ x + (1/2) xᵀ (∂²D/∂x²) x

where x = (Δx, Δy, Δσ)ᵀ
```

Set ∂D/∂x = 0 and solve:

```
x̂ = -(∂²D/∂x²)⁻¹ (∂D/∂x)
```

The refined keypoint location is the original candidate shifted by x̂. Discard if |D(x̂)| < threshold (low contrast) or if the ratio of principal curvatures > r (edge response).

**Orientation assignment**

Compute gradient magnitude and orientation for each pixel in a neighbourhood around the keypoint:

```
m(x, y) = sqrt[(L(x+1,y) - L(x-1,y))² + (L(x,y+1) - L(x,y-1))²]
θ(x, y) = atan2(L(x,y+1) - L(x,y-1),  L(x+1,y) - L(x-1,y))
```

Build a 36-bin orientation histogram weighted by m(x,y) and a Gaussian window. The dominant peak gives the keypoint orientation. This makes the descriptor rotation-invariant.

**Descriptor computation**

Rotate a 16×16 patch around the keypoint to the dominant orientation. Divide into a 4×4 grid of 4×4 cells. In each cell, compute an 8-bin gradient orientation histogram. Concatenate: 4×4×8 = 128 dimensions. Normalize to unit length, clamp values > 0.2, renormalize. This gives illumination invariance.

---

### Step 2 — Feature matching

For every pair of images (i, j), find putative correspondences by comparing SIFT descriptors.

**Nearest-neighbour search**

For descriptor `d_a` in image i, find its nearest (`d_1`) and second-nearest (`d_2`) in image j using L2 distance:

```
||d_a - d_1||₂ < ratio · ||d_a - d_2||₂
```

Lowe's ratio test with ratio=0.8. A match is accepted only if the nearest neighbour is significantly closer than the second-nearest. This rejects ambiguous matches where two database descriptors are similarly close.

**Exhaustive vs vocabulary tree matching**

COLMAP supports two strategies:
- Exhaustive: O(N²) pairs, compare all descriptors in all image pairs. Used for small datasets (<500 images).
- Vocabulary tree: Cluster all descriptors into a hierarchical tree (k-means at each level). Each descriptor is assigned a visual word by traversing the tree. Images sharing many visual words are candidate match pairs. O(N log N).

---

### Step 3 — Geometric verification (RANSAC + Fundamental/Essential matrix)

Putative matches contain many outliers. Geometric verification finds the geometrically consistent subset.

**The epipolar constraint**

For a point `x` in image i and its correspondence `x'` in image j:

```
x'ᵀ F x = 0
```

where F ∈ ℝ³ˣ³ is the fundamental matrix (rank 2, 7 DOF). If camera intrinsics K are known:

```
x'ᵀ E x = 0,   where E = Kᵀ F K   (Essential matrix, 5 DOF)
```

E encodes the relative rotation R and translation t between the two cameras:

```
E = [t]× R

where [t]× is the skew-symmetric matrix of t:
[t]× = [ 0   -t_z  t_y ]
       [ t_z   0  -t_x ]
       [-t_y  t_x   0  ]
```

**RANSAC loop for F/E estimation**

```
for i in range(max_iterations):
    sample = random.sample(matches, 7)     # 7-point algorithm for F
    F_candidates = estimate_F(sample)      # up to 3 solutions
    for F in F_candidates:
        inliers = [m for m in matches if |x'ᵀ F x| < threshold]
    keep F with most inliers
```

Iteration count to achieve probability p that at least one sample is outlier-free:

```
N = log(1 - p) / log(1 - (1 - ε)^s)

where ε = outlier fraction, s = sample size (7 for F, 5 for E)
```

COLMAP defaults: p=0.9999, threshold=4px Sampson distance.

**Sampson distance** (first-order approximation to reprojection error, cheaper to compute):

```
d_S(x, x') = (x'ᵀ F x)² / [(Fx)_1² + (Fx)_2² + (Fᵀx')_1² + (Fᵀx')_2²]
```

**Recovering R, t from E**

SVD decompose E = UΣVᵀ. There are 4 candidate (R, t) pairs:

```
R₁ = U W Vᵀ,   t₁ =  U[:,2]
R₂ = U W Vᵀ,   t₂ = -U[:,2]
R₃ = U Wᵀ Vᵀ,  t₃ =  U[:,2]
R₄ = U Wᵀ Vᵀ,  t₄ = -U[:,2]

where W = [0 -1 0; 1 0 0; 0 0 1]
```

Pick the solution where the most triangulated points have positive depth in both cameras (cheirality check).

---

### Step 4 — Triangulation

Given two (or more) calibrated cameras with known poses, find the 3D point P that projects to observed 2D points x_i.

**The projection model**

```
λ x = K [R | t] X

where:
  X = (X, Y, Z, 1)ᵀ  — homogeneous 3D point
  x = (u, v, 1)ᵀ     — homogeneous 2D observation
  λ = depth (scale factor)
  K = camera intrinsics
  [R | t] = camera extrinsics (3×4)
```

Let P_i = K_i [R_i | t_i] be the 3×4 projection matrix for camera i. For two views:

```
x₁ × (P₁ X) = 0
x₂ × (P₂ X) = 0
```

Each cross-product gives 2 independent equations (the third is linearly dependent). Stack for N views:

```
A X = 0,  where A ∈ ℝ^(2N × 4)
```

Solve via SVD: X = last column of V in A = UΣVᵀ. This is the Direct Linear Transform (DLT).

**Optimal triangulation (minimising reprojection error)**

DLT is not optimal under noise. Minimise:

```
min_X  Σᵢ ||xᵢ - π(P_i X)||²

where π(y) = (y₁/y₃, y₂/y₃) is the perspective division
```

Solved iteratively (Levenberg-Marquardt) or in closed form for N=2 (Hartley-Sturm).

**Reprojection error**

For a 3D point X and its observed 2D point x_i in camera i:

```
e_i = x_i - π(P_i X)
reprojection_error = ||e_i||₂  (pixels)
```

COLMAP discards triangulated points with reprojection error > 4px.

---

### Step 5 — Bundle Adjustment

Bundle Adjustment (BA) is the core optimization of SfM. Jointly refine all camera poses {R_i, t_i}, intrinsics {K_i}, and 3D points {X_j} to minimise total reprojection error:

```
min_{R,t,K,X}  Σᵢ Σⱼ  w_ij · ρ(||x_ij - π(P_i X_j)||²)

where:
  w_ij = 1 if point j is visible in image i, 0 otherwise
  ρ = robust loss (Huber or Cauchy) to downweight outliers
  π = perspective projection
```

This is a nonlinear least squares problem. COLMAP solves it with Ceres Solver using the Levenberg-Marquardt (LM) algorithm.

**LM iteration**

At each step, linearise the residuals r(θ) around current estimate θ:

```
r(θ + Δθ) ≈ r(θ) + J Δθ

where J = ∂r/∂θ  (Jacobian)
```

Solve the normal equations for Δθ:

```
(JᵀJ + λ I) Δθ = -Jᵀ r

where λ is the damping parameter (large λ → gradient descent; small λ → Gauss-Newton)
```

**Schur complement trick for efficiency**

The Jacobian J has a block-sparse structure: each residual depends on one camera and one point, never on two cameras or two points simultaneously. This gives JᵀJ a block-arrow structure. Use the Schur complement to eliminate point variables and solve only for camera variables first (much smaller system), then back-substitute for points.

```
[B  E ] [Δc]   [-b₁]
[Eᵀ C ] [Δp] = [-b₂]

Schur complement:  (B - E C⁻¹ Eᵀ) Δc = -b₁ + E C⁻¹ b₂
```

where B = camera-camera block, C = point-point block (diagonal), E = camera-point coupling.

C⁻¹ is trivial because C is block-diagonal (each 3D point only appears in its own block).

---

### Step 6 — Incremental SfM strategy

COLMAP builds the reconstruction incrementally, not all at once:

1. Find the best initial image pair: highest number of geometrically-verified matches, baseline not too small (parallax needed for triangulation), not too large (many inliers needed).
2. Triangulate 3D points from the initial pair.
3. Register a new image: find its 2D–3D correspondences (2D keypoints matched to already-triangulated 3D points), solve PnP to get the camera pose, run local BA.
4. Triangulate new 3D points visible in the new camera.
5. Run global BA every K new images to prevent drift.
6. Repeat from step 3.

**PnP (Perspective-n-Point)** — Step 3 above: given N known 3D points {X_j} and their 2D observations {x_j} in a new image with known K, recover R and t.

EPnP (used by COLMAP and our `localize.py`) expresses each 3D point as a weighted sum of 4 virtual control points:

```
X_j = Σₖ αⱼₖ c_k,   Σₖ αⱼₖ = 1
```

The control points are chosen as the centroid + 3 principal directions of the 3D point set. The 2D projections give linear constraints on the control point coordinates in camera space. Stack into a 12×12 linear system and solve via SVD. Closed-form O(N) complexity.

---

## COLMAP — Multi-View Stereo (MVS)

MVS takes the sparse SfM reconstruction as input and produces a dense point cloud or depth map per image.

### Patch Match Stereo

COLMAP's MVS uses PatchMatch, which initialises per-pixel depth and normal hypotheses randomly and iteratively propagates good hypotheses to neighbours.

**Depth map estimation for a reference image r**

For each pixel p in image r, we want to find depth d_p and surface normal n_p such that the patch around p has maximum photo-consistency with its projections in source images {s}.

**Photo-consistency: Normalised Cross-Correlation (NCC)**

For a pixel p in reference image r and its projection p'_s in source image s at depth d:

```
p'_s = π(K_s [R_s | t_s] π⁻¹(p, d, K_r, R_r, t_r))
```

where π⁻¹(p, d, K, R, t) back-projects pixel p at depth d into 3D world coords.

NCC score over a w×w patch:

```
NCC(p, p'_s) = Σ_{q∈W(p)} (I_r(q) - μ_r)(I_s(H·q) - μ_s) / (σ_r · σ_s · |W|)

where H is the homography induced by the depth-normal hypothesis (d, n)
```

The homography H maps patches between views accounting for surface orientation:

```
H = K_s (R_s - t_s nᵀ/d) Rᵣᵀ K_r⁻¹
```

**PatchMatch iteration**

```
for each pixel p:
    # Spatial propagation: try neighbours' hypotheses
    (d, n) = best of {current, left-neighbour, upper-neighbour}
    
    # Random search: perturb current hypothesis
    for i in range(max_search):
        d_new = d + δ_d · 2^(-i)   # halving search radius
        n_new = perturb(n, δ_n · 2^(-i))
        if photoconsistency(d_new, n_new) > photoconsistency(d, n):
            (d, n) = (d_new, n_new)
```

Alternate between forward (top-left to bottom-right) and backward passes. Converges in ~5 iterations.

**Multi-view aggregation**

Instead of pairwise NCC, COLMAP computes the geometric consistency score across all source views and uses the median depth to reject occluded hypotheses.

**Depth map fusion**

Individual per-image depth maps are fused into a global point cloud. For each depth estimate (p, d), back-project to 3D. Accept if the point is consistent (within threshold) with at least M other depth maps that observe it. This removes floating artifacts and strengthens accurate estimates.

---

## PnP + RANSAC — used in `localize.py`

Given N 2D–3D correspondences {(x_i, X_i)}, find R and t such that:

```
λ_i x_i = K (R X_i + t)   for all i
```

**EPnP (Efficient PnP)** — what `cv2.SOLVEPNP_EPNP` implements:

Express each 3D world point as a weighted combination of 4 control points {c_j}:

```
X_i = Σⱼ α_ij c_j,   with Σⱼ α_ij = 1  (barycentric coordinates)
```

Control points chosen as: c_0 = centroid of {X_i}, c_1,c_2,c_3 = centroid ± principal components (from PCA of {X_i}).

The unknown camera-frame coordinates of control points {c_j^c} satisfy:

```
x_i = π(Σⱼ α_ij c_j^c) = π(Σⱼ α_ij [c_jx^c, c_jy^c, c_jz^c]ᵀ)
```

Each observation gives 2 linear equations in the 12 unknowns (4 control points × 3 coords). Stack N points → M x = 0, M ∈ ℝ^(2N×12). Solve via SVD: solution lies in null space of M. For N≥6 the null space is 1-dimensional; for N<6 use additional constraints from the known distances between control points (they must be rigid).

Once {c_j^c} found, recover R and t by:
```
R, t from: [c₁^c - c₀^c | c₂^c - c₀^c | c₃^c - c₀^c] = R [c₁ - c₀ | c₂ - c₀ | c₃ - c₀]
→ solved by Procrustes / SVD
```

**RANSAC wrapper**

```
best_inliers = []
for i in range(max_iters):
    sample = random.sample(correspondences, 6)   # min for EPnP
    R, t = EPnP(sample)
    inliers = [j for j in all if reprojection_error(j, R, t) < threshold]
    if len(inliers) > len(best_inliers):
        best_inliers = inliers
        best_R, best_t = R, t

# Refine with all inliers (non-linear LM)
R, t = refine_LM(best_inliers, best_R, best_t)
```

Reprojection error threshold in our code: `--ransac_thresh 12.0` pixels. Tighter (5px) is more accurate but requires more inliers to succeed.

---

## Key papers

| Topic | Paper |
|---|---|
| SIFT | Lowe, "Distinctive image features from scale-invariant keypoints", IJCV 2004 |
| COLMAP SfM | Schönberger & Frahm, "Structure-from-Motion Revisited", CVPR 2016 |
| COLMAP MVS | Schönberger et al., "Pixelwise View Selection for Unstructured Multi-View Stereo", ECCV 2016 |
| EPnP | Lepetit, Moreno-Noguer & Fua, "EPnP: An Accurate O(n) Solution to the PnP Problem", IJCV 2009 |
| Bundle Adjustment | Triggs et al., "Bundle Adjustment — A Modern Synthesis", ICCV 1999 |
| LoFTR | Sun et al., "LoFTR: Detector-Free Local Feature Matching with Transformers", CVPR 2021 |
| OnePose++ | He et al., "OnePose++: Keypoint-Free One-Shot Object Pose Estimation without CAD Models", NeurIPS 2022 |
| SuperPoint | DeTone et al., "SuperPoint: Self-Supervised Interest Point Detection and Description", CVPRW 2018 |
| SuperGlue | Sarlin et al., "SuperGlue: Learning Feature Matching with Graph Neural Networks", CVPR 2020 |
| HLoc | Sarlin et al., "From Coarse to Fine: Robust Hierarchical Localization at Large Scale", CVPR 2019 |
| PatchMatch | Bleyer et al., "PatchMatch Stereo", BMVC 2011 |
| RANSAC | Fischler & Bolles, "Random Sample Consensus", Commun. ACM 1981 |

---

## Useful codebases

| Repo | What it gives you |
|---|---|
| `https://github.com/colmap/colmap` | COLMAP source — read `src/sfm/` for incremental SfM and `src/mvs/` for PatchMatch |
| `https://github.com/zju3dv/OnePose_Plus_Plus` | OnePose++ reference implementation — LoFTR-based SfM + 2D-3D matching network |
| `https://github.com/cvg/Hierarchical-Localization` | HLoc — coarse-to-fine visual localization, good reference for pose-guided search |
| `https://github.com/kornia/kornia` | Differentiable computer vision — GPU SIFT, LoFTR, geometric transforms |
| `https://github.com/mihaidusmanu/d2-net` | D2-Net — detect-and-describe jointly, stronger than SIFT on low-texture |
| `https://github.com/magicleap/SuperGluePretrainedNetwork` | SuperPoint + SuperGlue — current strong baseline for learned matching |
| `https://github.com/fabio-sim/LightGlue-ONNX` | LightGlue ONNX export — fast learned matching deployable on edge |
| `https://github.com/open3d/open3d` | Open3D — statistical outlier removal, point cloud visualisation |