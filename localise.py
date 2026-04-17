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
                0: 3,
                1: 4,
                2: 4,
                3: 5,
                4: 8,
            }

            num_params = model_params[model_id]
            params = struct.unpack("<" + "d"*num_params, f.read(8*num_params))

            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params
            }

    return cameras


def get_intrinsics(cam):
    model_id = cam["model_id"]
    p = cam["params"]

    if model_id == 0:
        f, cx, cy = p
        fx = fy = f
    elif model_id == 1:
        fx, fy, cx, cy = p
    elif model_id == 2:
        f, cx, cy, _ = p
        fx = fy = f
    elif model_id == 3:
        f, cx, cy, _, _ = p
        fx = fy = f
    elif model_id == 4:
        fx, fy, cx, cy = p[:4]
    else:
        raise ValueError("Unsupported camera model")

    return fx, fy, cx, cy


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
        if D[i, 0] < ratio * D[i, 1]:
            good_q.append(i)
            good_db.append(I[i, 0])

    return np.array(good_q), np.array(good_db)


# =============================
# -------- PNP ---------------
# =============================

def solve_pnp(kps, xyzs, idx_q, idx_db, K, thresh, iters):
    if len(idx_q) < 6:
        return None

    pts2d = kps[idx_q]
    pts3d = xyzs[idx_db]

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d,
        pts2d,
        K,
        None,
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

def draw_axes(image, K, R, t, scale=0.5):
    axis = np.float32([
        [0, 0, 0],
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, scale]
    ])

    rvec, _ = cv2.Rodrigues(R)
    imgpts, _ = cv2.projectPoints(axis, rvec, t, K, None)

    imgpts = imgpts.astype(int).reshape(-1, 2)

    origin = tuple(imgpts[0])
    x = tuple(imgpts[1])
    y = tuple(imgpts[2])
    z = tuple(imgpts[3])

    cv2.line(image, origin, x, (0, 0, 255), 3)   # X - red
    cv2.line(image, origin, y, (0, 255, 0), 3)   # Y - green
    cv2.line(image, origin, z, (255, 0, 0), 3)   # Z - blue

    return image


# =============================
# -------- MAIN ---------------
# =============================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--map_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--cameras_path", required=True)

    parser.add_argument("--output_path", default="output.jpg")

    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac_thresh", type=float, default=5.0)
    parser.add_argument("--ransac_iters", type=int, default=1000)

    parser.add_argument("--use_gpu", action="store_true")

    args = parser.parse_args()

    # -------------------------
    # Load map
    # -------------------------
    data = np.load(args.map_path)
    xyzs = data["xyzs"].astype(np.float32)
    descs = data["descs"].astype(np.float32)

    print(f"Loaded map: {xyzs.shape[0]} points")

    # -------------------------
    # FAISS index
    # -------------------------
    d = descs.shape[1]
    index = faiss.IndexFlatL2(d)

    if args.use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(descs)

    # -------------------------
    # Intrinsics
    # -------------------------
    cams = read_cameras_binary(args.cameras_path)
    cam = list(cams.values())[0]

    fx, fy, cx, cy = get_intrinsics(cam)

    print(f"Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # -------------------------
    # Load image
    # -------------------------
    img_gray = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(args.image_path)

    if img_gray is None:
        raise ValueError("Image not found")

    # -------------------------
    # Features
    # -------------------------
    kps, q_desc = extract_sift(img_gray)

    if q_desc is None:
        print("No features found")
        return

    print(f"Features: {len(kps)}")

    # -------------------------
    # Matching
    # -------------------------
    idx_q, idx_db = match_2d3d(q_desc, index, args.ratio)

    print(f"Matches: {len(idx_q)}")

    # -------------------------
    # PnP
    # -------------------------
    result = solve_pnp(
        kps, xyzs, idx_q, idx_db,
        K,
        args.ransac_thresh,
        args.ransac_iters
    )

    if result is None:
        print("PnP failed")
        return

    R, t, inliers = result

    print("\n=== Pose ===")
    print("R:\n", R)
    print("t:\n", t.flatten())
    print(f"Inliers: {len(inliers)} / {len(idx_q)}")

    # -------------------------
    # Visualization
    # -------------------------
    vis = draw_axes(img_color.copy(), K, R, t)

    # overlay text
    cv2.putText(
        vis,
        f"Inliers: {len(inliers)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # -------------------------
    # Save output
    # -------------------------
    cv2.imwrite(args.output_path, vis)
    print(f"Saved output to {args.output_path}")


if __name__ == "__main__":
    main()
