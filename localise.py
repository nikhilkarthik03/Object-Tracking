import argparse
import numpy as np
import cv2
import faiss
import struct


# =============================
# -------- CAMERA IO ----------
# =============================

def read_cameras_binary(path):
    cameras = {}

    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            model_params = {
                0: 3,   # SIMPLE_PINHOLE:  f, cx, cy
                1: 4,   # PINHOLE:         fx, fy, cx, cy
                2: 4,   # SIMPLE_RADIAL:   f, cx, cy, k1
                3: 5,   # RADIAL:          f, cx, cy, k1, k2
                4: 8,   # OPENCV:          fx, fy, cx, cy, k1, k2, p1, p2
            }

            num_params = model_params[model_id]
            params = struct.unpack("<" + "d" * num_params, f.read(8 * num_params))

            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params
            }

    return cameras


def get_intrinsics(cam):
    """
    FIX (Bug 4): Extract distortion coefficients from COLMAP camera params
    instead of always passing None. Returns (fx, fy, cx, cy, dist_coeffs)
    where dist_coeffs is a numpy array shaped (4,) or (5,) matching OpenCV
    convention: [k1, k2, p1, p2] or [k1, k2, p1, p2, k3].

    Previously this function silently discarded radial distortion, causing
    solvePnPRansac and projectPoints to reproject incorrectly for real lenses.
    """
    model_id = cam["model_id"]
    p = cam["params"]

    if model_id == 0:
        # SIMPLE_PINHOLE: f, cx, cy  — no distortion
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        dist = np.zeros(4, dtype=np.float64)

    elif model_id == 1:
        # PINHOLE: fx, fy, cx, cy  — no distortion
        fx, fy, cx, cy = p
        dist = np.zeros(4, dtype=np.float64)

    elif model_id == 2:
        # SIMPLE_RADIAL: f, cx, cy, k1
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        dist = np.array([p[3], 0.0, 0.0, 0.0], dtype=np.float64)

    elif model_id == 3:
        # RADIAL: f, cx, cy, k1, k2
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        dist = np.array([p[3], p[4], 0.0, 0.0], dtype=np.float64)

    elif model_id == 4:
        # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy, cx, cy = p[:4]
        dist = np.array(p[4:8], dtype=np.float64)

    else:
        raise ValueError(f"Unsupported camera model id: {model_id}")

    return fx, fy, cx, cy, dist


# =============================
# -------- FEATURES -----------
# =============================

def extract_sift(image):
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(image, None)

    if desc is None:
        return None, None

    desc = desc.astype(np.float32)
    desc /= (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-8)

    pts = np.array([kp.pt for kp in kps], dtype=np.float32)

    return pts, desc


# =============================
# -------- MATCHING -----------
# =============================

def match_2d3d(query_desc, index, ratio):
    D, I = index.search(query_desc, 2)

    good_q = []
    good_db = []

    for i in range(len(query_desc)):
        if D[i, 1] > 1e-8 and D[i, 0] < ratio * D[i, 1]:
            good_q.append(i)
            good_db.append(I[i, 0])

    return np.array(good_q), np.array(good_db)


# =============================
# -------- PNP ---------------
# =============================

def solve_pnp(kps, xyzs, idx_q, idx_db, K, dist_coeffs, thresh, iters):
    if len(idx_q) < 6:
        print(f"Too few matches for PnP: {len(idx_q)} (need >= 6)")
        return None

    # OpenCV solvePnPRansac signature:
    #   (objectPoints, imagePoints, cameraMatrix, distCoeffs, ...)
    # objectPoints = 3D world coords  (N, 3)  float32
    # imagePoints  = 2D pixel coords  (N, 2)  float32
    pts3d = xyzs[idx_db].reshape(-1, 1, 3).astype(np.float32)
    pts2d = kps[idx_q].reshape(-1, 1, 2).astype(np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d,                          # 3D object points  (correct order)
        pts2d,                          # 2D image points   (correct order)
        K,
        dist_coeffs,                    # FIX (Bug 4): pass real distortion, not None
        reprojectionError=thresh,
        iterationsCount=iters,
        confidence=0.99,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success or inliers is None:
        return None

    R, _ = cv2.Rodrigues(rvec)

    return R, tvec, inliers.flatten()


# =============================
# -------- VISUALIZATION ------
# =============================

def draw_axes(image, K, dist_coeffs, R, t, center, scale=0.5):
    """
    FIX (Bug 3): Draw the gimbal (coordinate axes) centred on `center`,
    which should be the centroid of the 3D map points (i.e. the object centre).

    Previously the axis origin was hard-coded to [0, 0, 0] — the SfM world
    origin — which is an arbitrary point in space, almost never at the object.
    The gimbal would appear in the wrong place or off-screen entirely.

    Args:
        center: (3,) float32 array — 3D world position to draw the gimbal at.
                Pass xyzs.mean(axis=0) from the loaded map.
        scale:  Length of each axis arm in world units.
    """
    cx, cy, cz = center.flatten()

    # Axis endpoints in world space, all relative to the object centre
    axis = np.float32([
        [cx,         cy,         cz        ],   # origin
        [cx + scale, cy,         cz        ],   # +X  red
        [cx,         cy + scale, cz        ],   # +Y  green
        [cx,         cy,         cz + scale],   # +Z  blue
    ])

    rvec, _ = cv2.Rodrigues(R)

    # FIX (Bug 4): use real dist_coeffs here too so projection is accurate
    imgpts, _ = cv2.projectPoints(axis, rvec, t, K, dist_coeffs)
    imgpts = imgpts.astype(int).reshape(-1, 2)

    origin = tuple(imgpts[0])
    x_end  = tuple(imgpts[1])
    y_end  = tuple(imgpts[2])
    z_end  = tuple(imgpts[3])

    # Draw thick axes with filled circle at origin
    cv2.line(image, origin, x_end, (0,   0,   255), 3)   # X — red
    cv2.line(image, origin, y_end, (0,   255, 0  ), 3)   # Y — green
    cv2.line(image, origin, z_end, (255, 0,   0  ), 3)   # Z — blue
    cv2.circle(image, origin, 6, (255, 255, 255), -1)     # white dot at gimbal centre

    # Small axis labels near tips
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "X", x_end, font, 0.6, (0,   0,   255), 2)
    cv2.putText(image, "Y", y_end, font, 0.6, (0,   255, 0  ), 2)
    cv2.putText(image, "Z", z_end, font, 0.6, (255, 0,   0  ), 2)

    return image


# =============================
# -------- MAIN ---------------
# =============================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--map_path",      required=True,
                        help="Path to map.npz built by build_map.py")
    parser.add_argument("--image_path",    required=True,
                        help="Query image to localize")
    parser.add_argument("--cameras_path",  required=True,
                        help="Path to COLMAP cameras.bin")

    parser.add_argument("--output_path",   default="output.jpg")

    parser.add_argument("--ratio",         type=float, default=0.75,
                        help="Lowe ratio test threshold")
    parser.add_argument("--ransac_thresh", type=float, default=5.0,
                        help="PnP RANSAC reprojection error threshold (pixels)")
    parser.add_argument("--ransac_iters",  type=int,   default=1000)
    parser.add_argument("--gimbal_scale",  type=float, default=0.3,
                        help="Length of gimbal arms in world units. "
                             "Tune this to match your scene scale.")

    parser.add_argument("--use_gpu",       action="store_true")

    args = parser.parse_args()

    # -------------------------
    # Load map
    # -------------------------
    data = np.load(args.map_path)
    xyzs  = data["xyzs"].astype(np.float32)   # (M, 3)
    descs = data["descs"].astype(np.float32)   # (M, 128)

    print(f"Loaded map: {xyzs.shape[0]} descriptor entries "
          f"(may be > # unique points due to multi-descriptor storage)")

    # FIX (Bug 3): compute object centroid ONCE from all map points.
    # This is the 3D position where the gimbal will be drawn.
    object_center = xyzs.mean(axis=0)
    print(f"Object centroid (gimbal origin): {object_center}")

    # -------------------------
    # FAISS index
    # -------------------------
    d = descs.shape[1]
    index = faiss.IndexFlatL2(d)

    if args.use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(descs)
    print(f"FAISS index built with {index.ntotal} vectors")

    # -------------------------
    # Intrinsics + distortion
    # -------------------------
    cams = read_cameras_binary(args.cameras_path)
    cam  = list(cams.values())[0]

    # FIX (Bug 4): unpack dist_coeffs instead of discarding them
    fx, fy, cx, cy, dist_coeffs = get_intrinsics(cam)

    print(f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"Distortion: {dist_coeffs}")

    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float32)

    # -------------------------
    # Load image
    # -------------------------
    img_gray  = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(args.image_path)

    if img_gray is None:
        raise ValueError(f"Image not found: {args.image_path}")

    print(f"Image size: {img_color.shape[1]}x{img_color.shape[0]}")

    # -------------------------
    # Extract SIFT features
    # -------------------------
    kps, q_desc = extract_sift(img_gray)

    if q_desc is None:
        print("No SIFT features found in query image")
        return

    print(f"Query features: {len(kps)}")

    # -------------------------
    # 2D-3D matching
    # -------------------------
    idx_q, idx_db = match_2d3d(q_desc, index, args.ratio)

    print(f"Matches after ratio test: {len(idx_q)}")

    if len(idx_q) == 0:
        print("No matches found — check that the query image shows the same object "
              "and that the map was built from images of the same scene.")
        return

    # -------------------------
    # PnP pose estimation
    # -------------------------
    result = solve_pnp(
        kps, xyzs, idx_q, idx_db,
        K,
        dist_coeffs,        # FIX (Bug 4): pass real distortion
        args.ransac_thresh,
        args.ransac_iters,
    )

    if result is None:
        print("PnP RANSAC failed — not enough inliers. "
              "Try lowering --ransac_thresh or --ratio, or rebuild the map.")
        return

    R, t, inliers = result

    print("\n=== Estimated Pose ===")
    print("R:\n", R)
    print("t:\n", t.flatten())
    print(f"Inliers: {len(inliers)} / {len(idx_q)}")

    # Camera position in world coords: C = -R^T @ t
    cam_pos = (-R.T @ t).flatten()
    print(f"Camera position in world: {cam_pos}")

    # -------------------------
    # Draw gimbal at object center
    # -------------------------
    vis = draw_axes(
        img_color.copy(),
        K,
        dist_coeffs,             # FIX (Bug 4)
        R,
        t,
        center=object_center,    # FIX (Bug 3): gimbal at object centroid, not [0,0,0]
        scale=args.gimbal_scale,
    )

    # Overlay stats
    cv2.putText(vis, f"Inliers: {len(inliers)}/{len(idx_q)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(vis, f"Cam: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # -------------------------
    # Save
    # -------------------------
    cv2.imwrite(args.output_path, vis)
    print(f"\nSaved output to {args.output_path}")


if __name__ == "__main__":
    main()