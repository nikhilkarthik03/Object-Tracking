# Resources

## COLMAP — Structure from Motion (SfM)

COLMAP implements incremental SfM. The goal is: given $N$ unordered images, simultaneously recover the 3D structure of the scene and the 6-DoF camera pose for every image.

---

### Step 1 — Feature Extraction (SIFT)

#### Scale-space construction

Build a Gaussian scale space by convolving image $I$ with Gaussians at increasing $\sigma$:

$$L(x, y, \sigma) = G(x, y, \sigma) \ast I(x, y)$$

where the Gaussian kernel is:

$$G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

The **Difference of Gaussians** (DoG) approximates the Laplacian of Gaussian $\sigma^2 \nabla^2 G$ and is far cheaper to compute:

$$D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma)$$

Keypoints are local extrema of $D$ across both space $(x,y)$ and scale $\sigma$. A candidate $(x, y, \sigma)$ is accepted if $D(x,y,\sigma)$ is greater (or less) than all 26 neighbours in its $3 \times 3 \times 3$ spatial-scale neighbourhood.

#### Sub-pixel refinement

Fit a 3D quadratic to $D$ around the candidate via a second-order Taylor expansion:

$$D(\mathbf{x}) \approx D + \frac{\partial D}{\partial \mathbf{x}}^\top \mathbf{x} + \frac{1}{2}\mathbf{x}^\top \frac{\partial^2 D}{\partial \mathbf{x}^2} \mathbf{x}, \qquad \mathbf{x} = (\Delta x,\, \Delta y,\, \Delta\sigma)^\top$$

Setting $\partial D / \partial \mathbf{x} = \mathbf{0}$ and solving:

$$\hat{\mathbf{x}} = -\left(\frac{\partial^2 D}{\partial \mathbf{x}^2}\right)^{-1} \frac{\partial D}{\partial \mathbf{x}}$$

The refined keypoint location is the original candidate shifted by $\hat{\mathbf{x}}$. Discard if $|D(\hat{\mathbf{x}})| < \tau_c$ (low contrast) or if the ratio of principal curvatures of $D$ exceeds $r$ (on an edge, not a corner):

$$\frac{(H_{11} + H_{22})^2}{H_{11}H_{22} - H_{12}^2} > \frac{(r+1)^2}{r}, \qquad H = \frac{\partial^2 D}{\partial \mathbf{x}^2}\bigg|_{xy}$$

COLMAP uses $r = 10$, $\tau_c = 0.01$.

#### Orientation assignment

For each pixel in a neighbourhood around the keypoint, compute gradient magnitude and orientation:

$$m(x,y) = \sqrt{\bigl[L(x{+}1,y) - L(x{-}1,y)\bigr]^2 + \bigl[L(x,y{+}1) - L(x,y{-}1)\bigr]^2}$$

$$
\theta(x,y) = \mathrm{atan2}\bigl(
L(x,y+1) - L(x,y-1),\ 
L(x+1,y) - L(x-1,y)
\bigr)
$$

Build a 36-bin orientation histogram weighted by $m(x,y)$ and a Gaussian window of scale $1.5\sigma$. The dominant peak direction becomes the canonical orientation, making the descriptor rotation-invariant.

#### Descriptor computation

Rotate a $16 \times 16$ patch to the canonical orientation. Divide into a $4 \times 4$ grid of $4 \times 4$ cells. In each cell, compute an 8-bin gradient orientation histogram. Concatenate all cells:

$$\mathbf{d} \in \mathbb{R}^{4 \times 4 \times 8} = \mathbb{R}^{128}$$

Normalise to unit length, clamp values $> 0.2$, renormalise. Clamping suppresses non-linear illumination effects.

---

### Step 2 — Feature Matching

For every image pair $(i, j)$, find putative correspondences by comparing SIFT descriptors.

#### Lowe's ratio test

For descriptor $\mathbf{d}_a$ in image $i$, find its nearest $\mathbf{d}_1$ and second-nearest $\mathbf{d}_2$ in image $j$ under the $\ell_2$ distance. Accept the match only if:

$$\|\mathbf{d}_a - \mathbf{d}_1\|_2 < r \cdot \|\mathbf{d}_a - \mathbf{d}_2\|_2, \qquad r = 0.8$$

This rejects ambiguous matches where two database descriptors are similarly close — the root cause of most false matches in low-texture regions.

#### Matching strategy

| Strategy | Complexity | When to use |
|---|---|---|
| Exhaustive | $O(N^2)$ image pairs | $N < 500$ images |
| Vocabulary tree | $O(N \log N)$ | $N \geq 500$ images |

The vocabulary tree clusters all descriptors via hierarchical $k$-means. Each descriptor is assigned a visual word by traversing the tree in $O(\log N)$. Images sharing many visual words are candidate match pairs.

---

### Step 3 — Geometric Verification (RANSAC + Epipolar Geometry)

Putative matches contain many outliers from the ratio test. Geometric verification finds the largest geometrically-consistent inlier set.

#### The epipolar constraint

For a point $\mathbf{x}$ in image $i$ and its correspondence $\mathbf{x}'$ in image $j$ (homogeneous 2D coordinates):

$${\mathbf{x}'}^\top \mathbf{F}\, \mathbf{x} = 0$$

where $\mathbf{F} \in \mathbb{R}^{3 \times 3}$ is the **fundamental matrix** (rank 2, 7 DOF). When camera intrinsics $\mathbf{K}$ are known, the **essential matrix** $\mathbf{E}$ (5 DOF) carries only the extrinsic information:

$${\mathbf{x}'}^\top \mathbf{E}\, \mathbf{x} = 0, \qquad \mathbf{E} = \mathbf{K}^\top \mathbf{F}\, \mathbf{K}$$

$\mathbf{E}$ decomposes into the relative rotation $\mathbf{R}$ and translation $\mathbf{t}$ between the two cameras:

$$\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$$

where $[\mathbf{t}]_\times$ is the skew-symmetric cross-product matrix:

$$[\mathbf{t}]_\times = \begin{pmatrix} 0 & -t_z & t_y \\ t_z & 0 & -t_x \\ -t_y & t_x & 0 \end{pmatrix}$$

#### Sampson distance

The algebraic error $|{\mathbf{x}'}^\top \mathbf{F}\mathbf{x}|$ is not in pixel units. The first-order geometric approximation (Sampson distance) is cheap to compute and used as the RANSAC inlier threshold:

$$d_S(\mathbf{x}, \mathbf{x}') = \frac{\bigl({\mathbf{x}'}^\top \mathbf{F}\mathbf{x}\bigr)^2}{(\mathbf{F}\mathbf{x})_1^2 + (\mathbf{F}\mathbf{x})_2^2 + (\mathbf{F}^\top\mathbf{x}')_1^2 + (\mathbf{F}^\top\mathbf{x}')_2^2}$$

COLMAP threshold: $d_S < 4$ px. Confidence: $p = 0.9999$.

#### RANSAC iteration count

To achieve probability $p$ that at least one sample of size $s$ is entirely free of outliers (outlier fraction $\varepsilon$):

$$N = \frac{\log(1 - p)}{\log\!\bigl(1 - (1-\varepsilon)^s\bigr)}$$

For $\mathbf{F}$: $s = 7$ (7-point algorithm, up to 3 solutions). For $\mathbf{E}$: $s = 5$ (5-point algorithm).

#### Recovering $\mathbf{R},\, \mathbf{t}$ from $\mathbf{E}$

SVD decompose $\mathbf{E} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$. There are 4 candidate solutions:

$$\mathbf{W} = \begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

$$(\mathbf{R}_1,\, \mathbf{t}_1) = \bigl(\mathbf{U}\mathbf{W}\mathbf{V}^\top,\ +\mathbf{u}_3\bigr), \qquad (\mathbf{R}_2,\, \mathbf{t}_2) = \bigl(\mathbf{U}\mathbf{W}\mathbf{V}^\top,\ -\mathbf{u}_3\bigr)$$

$$(\mathbf{R}_3,\, \mathbf{t}_3) = \bigl(\mathbf{U}\mathbf{W}^\top\mathbf{V}^\top,\ +\mathbf{u}_3\bigr), \qquad (\mathbf{R}_4,\, \mathbf{t}_4) = \bigl(\mathbf{U}\mathbf{W}^\top\mathbf{V}^\top,\ -\mathbf{u}_3\bigr)$$

where $\mathbf{u}_3$ is the third column of $\mathbf{U}$. The correct solution is the one where the most triangulated points have **positive depth** in both cameras (cheirality check).

---

### Step 4 — Triangulation

Given two or more calibrated cameras with known poses, find the 3D point $\mathbf{X}$ that projects to observed 2D points $\{\mathbf{x}_i\}$.

#### Projection model

$$\lambda_i\, \mathbf{x}_i = \mathbf{K}_i\,[\mathbf{R}_i \mid \mathbf{t}_i]\, \mathbf{X}$$

where $\mathbf{X} = (X, Y, Z, 1)^\top$ is the homogeneous 3D point, $\mathbf{x}_i = (u, v, 1)^\top$ is the homogeneous 2D observation, and $\lambda_i$ is an unknown depth scale. Let $\mathbf{P}_i = \mathbf{K}_i[\mathbf{R}_i \mid \mathbf{t}_i] \in \mathbb{R}^{3 \times 4}$ be the projection matrix.

#### Direct Linear Transform (DLT)

The cross-product $\mathbf{x}_i \times (\mathbf{P}_i \mathbf{X}) = \mathbf{0}$ eliminates $\lambda_i$ and gives 2 independent linear equations per view (the third is redundant):

$$\begin{pmatrix} \mathbf{p}_{i,3}^\top X\, u_i - \mathbf{p}_{i,1}^\top \mathbf{X} \\ \mathbf{p}_{i,3}^\top X\, v_i - \mathbf{p}_{i,2}^\top \mathbf{X} \end{pmatrix} = \mathbf{0}$$

Stack for $N$ views to get the homogeneous linear system:

$$\mathbf{A}\mathbf{X} = \mathbf{0}, \qquad \mathbf{A} \in \mathbb{R}^{2N \times 4}$$

Solve via SVD $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$: the solution is the last column of $\mathbf{V}$ (right singular vector corresponding to the smallest singular value).

#### Optimal triangulation

DLT minimises an algebraic error, not the reprojection error, so it is not statistically optimal under image noise. The correct objective is:

$$\min_{\mathbf{X}} \sum_i \left\|\mathbf{x}_i - \pi(\mathbf{P}_i \mathbf{X})\right\|^2$$

where $\pi(\mathbf{y}) = (y_1/y_3,\, y_2/y_3)^\top$ is the perspective division. Solved iteratively with Levenberg-Marquardt, or in closed form for $N=2$ via the Hartley-Sturm algorithm.

#### Reprojection error

$$e_i = \left\|\mathbf{x}_i - \pi(\mathbf{P}_i \mathbf{X})\right\|_2 \quad \text{(pixels)}$$

COLMAP discards triangulated points with $e_i > 4$ px in any view.

---

### Step 5 — Bundle Adjustment

Bundle Adjustment (BA) is the core optimisation of SfM. It jointly refines all camera poses $\{\mathbf{R}_i, \mathbf{t}_i\}$, intrinsics $\{\mathbf{K}_i\}$, and 3D points $\{\mathbf{X}_j\}$ to minimise total reprojection error:

$$\min_{\mathbf{R},\,\mathbf{t},\,\mathbf{K},\,\mathbf{X}} \sum_i \sum_j w_{ij}\; \rho\!\left(\left\|\mathbf{x}_{ij} - \pi(\mathbf{P}_i \mathbf{X}_j)\right\|^2\right)$$

where $w_{ij} = 1$ if point $j$ is visible in image $i$, $\rho$ is a robust loss (Huber or Cauchy) to downweight outliers, and $\pi$ is the perspective projection. COLMAP solves this with Ceres Solver using Levenberg-Marquardt (LM).

#### Levenberg-Marquardt iteration

Let $\boldsymbol{\theta}$ collect all unknowns (camera poses + 3D points). Linearise residuals $\mathbf{r}(\boldsymbol{\theta})$ around the current estimate:

$$\mathbf{r}(\boldsymbol{\theta} + \Delta\boldsymbol{\theta}) \approx \mathbf{r}(\boldsymbol{\theta}) + \mathbf{J}\,\Delta\boldsymbol{\theta}, \qquad \mathbf{J} = \frac{\partial \mathbf{r}}{\partial \boldsymbol{\theta}}$$

Solve the **normal equations** for the update $\Delta\boldsymbol{\theta}$:

$$\bigl(\mathbf{J}^\top \mathbf{J} + \lambda\, \mathbf{I}\bigr)\, \Delta\boldsymbol{\theta} = -\mathbf{J}^\top \mathbf{r}$$

The damping parameter $\lambda$ interpolates between Gauss-Newton ($\lambda \to 0$, fast near the optimum) and gradient descent ($\lambda \to \infty$, stable far from the optimum). $\lambda$ is increased on a bad step and decreased on a good step.

#### Schur complement trick

$\mathbf{J}^\top\mathbf{J}$ has block-sparse structure: each residual $r_{ij}$ depends only on camera $i$ and point $j$, never on two cameras or two points. Partition $\boldsymbol{\theta} = (\mathbf{c}, \mathbf{p})$ (cameras, points):

$$\begin{pmatrix} \mathbf{B} & \mathbf{E} \\ \mathbf{E}^\top & \mathbf{C} \end{pmatrix} \begin{pmatrix} \Delta\mathbf{c} \\ \Delta\mathbf{p} \end{pmatrix} = \begin{pmatrix} -\mathbf{b}_1 \\ -\mathbf{b}_2 \end{pmatrix}$$

where $\mathbf{B}$ is the camera-camera block, $\mathbf{C}$ is the point-point block (block-diagonal), and $\mathbf{E}$ is the camera-point coupling. Eliminate $\Delta\mathbf{p}$ via the **Schur complement**:

$$\underbrace{(\mathbf{B} - \mathbf{E}\mathbf{C}^{-1}\mathbf{E}^\top)}_{\text{Schur complement}} \Delta\mathbf{c} = -\mathbf{b}_1 + \mathbf{E}\mathbf{C}^{-1}\mathbf{b}_2$$

$\mathbf{C}^{-1}$ is trivial because $\mathbf{C}$ is block-diagonal (each 3D point has its own $3 \times 3$ block). Solving for $\Delta\mathbf{c}$ (camera unknowns only) is much cheaper than solving the full system. Then back-substitute to get $\Delta\mathbf{p}$.

---

### Step 6 — Incremental SfM Strategy

COLMAP builds the reconstruction incrementally:

1. **Seed pair:** find the image pair with the highest number of geometrically-verified matches, sufficient baseline (parallax), and high estimated homography inlier ratio.
2. **Initialise:** triangulate 3D points from the seed pair via DLT.
3. **Register:** for each new image, find 2D–3D correspondences to already-triangulated points, solve PnP (below) for its pose, run local BA.
4. **Triangulate:** create new 3D points visible in the new camera.
5. **Global BA:** every $K$ new images, run BA over all cameras and points to prevent drift accumulation.
6. **Repeat** from step 3.

#### PnP — registering a new camera

Given $N$ known 3D points $\{\mathbf{X}_j\}$ and their 2D observations $\{\mathbf{x}_j\}$ in a new image with known $\mathbf{K}$, recover $\mathbf{R}$ and $\mathbf{t}$.

**EPnP** (used in COLMAP and `localize.py`) expresses each 3D point as a weighted sum of 4 virtual control points $\{\mathbf{c}_k\}$:

$$\mathbf{X}_j = \sum_{k=0}^{3} \alpha_{jk}\, \mathbf{c}_k, \qquad \sum_{k=0}^{3} \alpha_{jk} = 1$$

Control points: $\mathbf{c}_0 = $ centroid of $\{\mathbf{X}_j\}$; $\mathbf{c}_1, \mathbf{c}_2, \mathbf{c}_3 = $ centroid $\pm$ principal components (PCA). The $\alpha_{jk}$ are the barycentric coordinates (computed analytically from the $\mathbf{X}_j$).

Let $\mathbf{c}_k^c$ be the unknown camera-frame coordinates of control point $k$. Each 2D observation $\mathbf{x}_j$ gives 2 linear equations in the 12 unknowns:

$$\sum_{k=0}^{3} \alpha_{jk}\, c_{k,3}^c \cdot u_j = \sum_{k=0}^{3} \alpha_{jk}\, c_{k,1}^c \cdot f_x + \alpha_{jk}\, c_{k,3}^c \cdot c_x$$

Stack $N$ points into $\mathbf{M} \in \mathbb{R}^{2N \times 12}$ and solve $\mathbf{M}\mathbf{v} = \mathbf{0}$ via SVD. For $N \geq 6$ the null space is 1-dimensional; for smaller $N$ enforce rigidity constraints (known inter-control-point distances). Recover $\mathbf{R}$ and $\mathbf{t}$ from the found $\{\mathbf{c}_k^c\}$ via Procrustes:

$$\bigl[\mathbf{c}_1^c - \mathbf{c}_0^c \;\big|\; \mathbf{c}_2^c - \mathbf{c}_0^c \;\big|\; \mathbf{c}_3^c - \mathbf{c}_0^c\bigr] = \mathbf{R}\,\bigl[\mathbf{c}_1 - \mathbf{c}_0 \;\big|\; \mathbf{c}_2 - \mathbf{c}_0 \;\big|\; \mathbf{c}_3 - \mathbf{c}_0\bigr]$$

EPnP runs in $O(N)$ — linear in the number of correspondences.

---

## COLMAP — Multi-View Stereo (MVS)

MVS takes the sparse SfM reconstruction as input and produces a dense depth map (and thus a dense point cloud) per image.

### PatchMatch Stereo

COLMAP MVS uses PatchMatch: initialise per-pixel depth $d_p$ and surface normal $\mathbf{n}_p$ randomly, then iteratively propagate good hypotheses to neighbours.

#### Photo-consistency: Normalised Cross-Correlation (NCC)

For pixel $\mathbf{p}$ in reference image $r$, project to source image $s$ at depth $d$:

$$\mathbf{p}'_s = \pi\!\left(\mathbf{K}_s\,[\mathbf{R}_s \mid \mathbf{t}_s]\,\pi^{-1}(\mathbf{p},\, d,\, \mathbf{K}_r,\, \mathbf{R}_r,\, \mathbf{t}_r)\right)$$

where $\pi^{-1}(\mathbf{p}, d, \cdot)$ back-projects pixel $\mathbf{p}$ at depth $d$ into world coordinates. NCC over a $w \times w$ patch $\mathcal{W}(\mathbf{p})$:

$$\operatorname{NCC}(\mathbf{p},\, \mathbf{p}'_s) = \frac{\displaystyle\sum_{\mathbf{q} \in \mathcal{W}(\mathbf{p})} \bigl(I_r(\mathbf{q}) - \mu_r\bigr)\bigl(I_s(\mathbf{H}\mathbf{q}) - \mu_s\bigr)}{\sigma_r\, \sigma_s\, |\mathcal{W}|}$$

The homography $\mathbf{H}$ maps the reference patch to the source view, accounting for surface orientation via the depth-normal hypothesis $(d, \mathbf{n})$:

$$\mathbf{H} = \mathbf{K}_s \left(\mathbf{R}_s - \frac{\mathbf{t}_s\, \mathbf{n}^\top}{d}\right) \mathbf{R}_r^\top\, \mathbf{K}_r^{-1}$$

This is the standard planar homography formula. When $\mathbf{n}$ is the true surface normal and $d$ the true depth, $\mathbf{H}$ correctly warps the patch to account for surface tilt.

#### PatchMatch iteration

```
for each pixel p (forward pass: top-left → bottom-right):
    # 1. Spatial propagation — try neighbours' hypotheses
    (d, n) = argmax NCC over {current, left-neighbour, top-neighbour}

    # 2. Random refinement — perturb current hypothesis
    for i = 0, 1, ..., max_search:
        d_new = d + δ_d · 2^(−i)          # halving search radius
        n_new = perturb(n, δ_n · 2^(−i))  # shrinking angular perturbation
        if NCC(d_new, n_new) > NCC(d, n):
            (d, n) = (d_new, n_new)

# Backward pass: bottom-right → top-left (same logic)
```

Alternating forward and backward passes converge in $\sim 5$ iterations. Good hypotheses propagate quickly across uniform regions.

#### Depth map fusion

Per-image depth maps are fused into a global point cloud. A depth estimate $(\mathbf{p}, d)$ is accepted if it is consistent (within a pixel threshold) with at least $M$ other depth maps that observe the same 3D point. Consistency is checked by reprojecting the back-projected 3D point into neighbouring views and comparing depths. This suppresses floaters and reinforces accurate estimates.

---

## PnP + RANSAC — used in `localize.py`

Given $N$ 2D–3D correspondences $\{(\mathbf{x}_i, \mathbf{X}_i)\}$, find $\mathbf{R}$ and $\mathbf{t}$ such that:

$$\lambda_i\, \mathbf{x}_i = \mathbf{K}\,(\mathbf{R}\,\mathbf{X}_i + \mathbf{t}) \quad \forall\, i$$

This is solved by wrapping EPnP (described above) in a RANSAC loop:

```
best_inliers = []
for i in range(max_iters):
    sample = random.sample(correspondences, 6)   # minimum for EPnP
    R, t   = EPnP(sample)
    inliers = [j for j in all_correspondences
               if reprojection_error(j, R, t) < threshold]
    if len(inliers) > len(best_inliers):
        best_inliers = inliers
        best_R, best_t = R, t

# Non-linear refinement over all inliers
R, t = Levenberg_Marquardt(best_inliers, best_R, best_t)
```

The reprojection error for correspondence $(\mathbf{x}_i, \mathbf{X}_i)$:

$$e_i = \left\|\mathbf{x}_i - \pi\!\bigl(\mathbf{K}(\mathbf{R}\,\mathbf{X}_i + \mathbf{t})\bigr)\right\|_2$$

In our code: `--ransac_thresh 12.0` px (relaxed for low-inlier scenarios). Tighter (5 px) gives more accurate poses but requires higher inlier count to succeed.

---

## Key Papers

| Topic | Citation |
|---|---|
| SIFT | Lowe, *Distinctive image features from scale-invariant keypoints*, IJCV 2004 |
| COLMAP SfM | Schönberger & Frahm, *Structure-from-Motion Revisited*, CVPR 2016 |
| COLMAP MVS | Schönberger et al., *Pixelwise View Selection for Unstructured Multi-View Stereo*, ECCV 2016 |
| EPnP | Lepetit, Moreno-Noguer & Fua, *EPnP: An Accurate O(n) Solution to the PnP Problem*, IJCV 2009 |
| Bundle Adjustment | Triggs et al., *Bundle Adjustment — A Modern Synthesis*, ICCV 1999 |
| LoFTR | Sun et al., *LoFTR: Detector-Free Local Feature Matching with Transformers*, CVPR 2021 |
| OnePose++ | He et al., *OnePose++: Keypoint-Free One-Shot Object Pose Estimation without CAD Models*, NeurIPS 2022 |
| SuperPoint | DeTone et al., *SuperPoint: Self-Supervised Interest Point Detection and Description*, CVPRW 2018 |
| SuperGlue | Sarlin et al., *SuperGlue: Learning Feature Matching with Graph Neural Networks*, CVPR 2020 |
| HLoc | Sarlin et al., *From Coarse to Fine: Robust Hierarchical Localization at Large Scale*, CVPR 2019 |
| PatchMatch | Bleyer et al., *PatchMatch Stereo*, BMVC 2011 |
| RANSAC | Fischler & Bolles, *Random Sample Consensus*, Commun. ACM 1981 |

---

## Useful Codebases

| Repo | What it gives you |
|---|---|
| [colmap/colmap](https://github.com/colmap/colmap) | COLMAP source — `src/sfm/` for incremental SfM, `src/mvs/` for PatchMatch |
| [zju3dv/OnePose_Plus_Plus](https://github.com/zju3dv/OnePose_Plus_Plus) | OnePose++ reference — LoFTR SfM + 2D-3D matching network |
| [cvg/Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization) | HLoc — coarse-to-fine visual localization, pose-guided search reference |
| [kornia/kornia](https://github.com/kornia/kornia) | Differentiable CV — GPU SIFT, LoFTR, geometric transforms |
| [mihaidusmanu/d2-net](https://github.com/mihaidusmanu/d2-net) | D2-Net — detect-and-describe jointly, stronger than SIFT on low-texture |
| [magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) | SuperPoint + SuperGlue — strong learned matching baseline |
| [fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) | LightGlue ONNX — fast learned matching for edge deployment |
| [open3d/open3d](https://github.com/open3d/open3d) | Open3D — statistical outlier removal, point cloud visualisation |