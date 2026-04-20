import argparse
import sqlite3
import numpy as np
import struct
import os


# =============================
# -------- IO UTILS ----------
# =============================

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path):
    points3D = {}

    with open(path, "rb") as f:
        num_points = read_next_bytes(f, 8, "Q")[0]

        for _ in range(num_points):
            pid = read_next_bytes(f, 8, "Q")[0]
            xyz = np.array(read_next_bytes(f, 24, "ddd"), dtype=np.float32)

            _ = read_next_bytes(f, 3, "BBB")
            _ = read_next_bytes(f, 8, "d")[0]

            track_len = read_next_bytes(f, 8, "Q")[0]

            track = []
            for _ in range(track_len):
                image_id = read_next_bytes(f, 4, "I")[0]
                kp_idx = read_next_bytes(f, 4, "I")[0]
                track.append((image_id, kp_idx))

            points3D[pid] = {
                "xyz": xyz,
                "track": track
            }

    return points3D


def load_descriptors(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT image_id, rows, cols, data FROM descriptors")

    descriptors = {}

    for image_id, rows, cols, data in cursor:
        desc = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols)
        descriptors[image_id] = desc.astype(np.float32)

    conn.close()
    return descriptors


# =============================
# -------- MAP BUILD ----------
# =============================

class MapPoint:
    def __init__(self, xyz):
        self.xyz = xyz
        self.descs = []


def build_map(points3D, descriptors, min_views):
    map_points = {}

    for pid, pdata in points3D.items():
        track = pdata["track"]

        if len(track) < min_views:
            continue

        mp = MapPoint(pdata["xyz"])

        for (img_id, kp_idx) in track:
            if img_id not in descriptors:
                continue

            descs = descriptors[img_id]

            if kp_idx < len(descs):
                mp.descs.append(descs[kp_idx])

        if len(mp.descs) > 0:
            map_points[pid] = mp

    return map_points


def select_diverse_descriptors(descs_list, max_k=4):
    """
    FIX (Bug 1): Instead of collapsing all observations into one mean vector
    (which loses viewpoint diversity), we keep up to max_k descriptors that
    are maximally spread from each other via greedy farthest-point selection.

    This means a 3D point seen from very different angles will have multiple
    representatives in the index, dramatically improving match recall.
    """
    descs = np.array(descs_list, dtype=np.float32)

    if len(descs) <= max_k:
        return descs

    # Greedy farthest-point sampling
    selected = [0]
    for _ in range(max_k - 1):
        # Distance from each descriptor to the nearest already-selected one
        dists = np.array([
            min(np.linalg.norm(descs[i] - descs[s]) for s in selected)
            for i in range(len(descs))
            if i not in selected
        ])
        remaining = [i for i in range(len(descs)) if i not in selected]
        selected.append(remaining[np.argmax(dists)])

    return descs[selected]


def aggregate_descriptors(map_points, normalize=True, max_k=4):
    """
    FIX (Bug 1): Return one row per (point, descriptor) pair instead of one
    row per point. The returned point_ids array maps each descriptor row back
    to its 3D point so localize.py can look up xyzs[idx_db] correctly.
    """
    all_descs = []
    all_xyzs = []
    all_pids = []

    for pid, mp in map_points.items():
        diverse = select_diverse_descriptors(mp.descs, max_k=max_k)

        for d in diverse:
            if normalize:
                d = d / (np.linalg.norm(d) + 1e-8)
            all_descs.append(d.astype(np.float32))
            all_xyzs.append(mp.xyz.astype(np.float32))
            all_pids.append(pid)

    return (
        np.vstack(all_descs),   # (M, 128)  — M >= N (multiple descs per point)
        np.vstack(all_xyzs),    # (M, 3)    — xyz repeated for each descriptor
    )


# =============================
# -------- MAIN ---------------
# =============================

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Build 3D map from COLMAP outputs")

    parser.add_argument("--sparse_path", required=True,
                        help="Path to COLMAP sparse folder (e.g. sparse/0)")
    parser.add_argument("--database_path", required=True,
                        help="Path to COLMAP database.db")
    parser.add_argument("--output_path", required=True,
                        help="Output .npz file")
    parser.add_argument("--min_views", type=int, default=3,
                        help="Minimum observations per 3D point (raised from 2 to 3)")
    parser.add_argument("--max_descs_per_point", type=int, default=4,
                        help="Max diverse descriptors to keep per 3D point")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Disable descriptor normalization")

    args = parser.parse_args()

    points_path = os.path.join(args.sparse_path, "points3D.bin")

    print("Loading points3D...")
    points3D = read_points3D_binary(points_path)

    print("Loading descriptors...")
    descriptors = load_descriptors(args.database_path)

    print("Building map...")
    map_points = build_map(points3D, descriptors, args.min_views)

    print(f"Valid map points: {len(map_points)}")

    print("Aggregating descriptors (diverse multi-descriptor per point)...")
    descs, xyzs = aggregate_descriptors(
        map_points,
        normalize=not args.no_normalize,
        max_k=args.max_descs_per_point,
    )

    print("Saving output...")
    np.savez_compressed(args.output_path, xyzs=xyzs, descs=descs)

    print(f"Saved to {args.output_path}")
    print(f"xyzs shape: {xyzs.shape}")
    print(f"descs shape: {descs.shape}")
    print(f"(Multiple rows per 3D point — each descriptor indexes back to its xyz)")


if __name__ == "__main__":
    main()