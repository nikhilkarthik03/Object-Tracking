from plyfile import PlyData
import argparse
import sqlite3
import numpy as np
import struct
import os

from scipy.spatial import cKDTree


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
# -------- PLY LOADING --------
# =============================

def load_ply_xyz(path):
    """
    Load XYZ points from a PLY file (ASCII or binary).
    Returns: (N, 3) numpy array
    """
    ply = PlyData.read(path)

    if 'vertex' not in ply:
        raise ValueError(f"{path} does not contain vertex data")

    v = ply['vertex']

    # Validate fields
    fields = v.data.dtype.names
    for key in ('x', 'y', 'z'):
        if key not in fields:
            raise ValueError(f"Missing '{key}' in PLY vertex fields: {fields}")

    xyz = np.vstack([v['x'], v['y'], v['z']]).T
    return xyz

def build_spatial_filter(pruned_xyz):
    return cKDTree(pruned_xyz)


# =============================
# -------- MAP BUILD ----------
# =============================

class MapPoint:
    def __init__(self, xyz):
        self.xyz = xyz
        self.descs = []


def build_map(points3D, descriptors, min_views, tree=None, radius=None):
    map_points = {}

    for pid, pdata in points3D.items():
        xyz = pdata["xyz"]

        # --- NEW: spatial filtering ---
        if tree is not None:
            dist, _ = tree.query(xyz, k=1)
            if dist > radius:
                continue
        # --------------------------------

        track = pdata["track"]

        if len(track) < min_views:
            continue

        mp = MapPoint(xyz)

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
    descs = np.array(descs_list, dtype=np.float32)

    if len(descs) <= max_k:
        return descs

    selected = [0]
    for _ in range(max_k - 1):
        dists = np.array([
            min(np.linalg.norm(descs[i] - descs[s]) for s in selected)
            for i in range(len(descs))
            if i not in selected
        ])
        remaining = [i for i in range(len(descs)) if i not in selected]
        selected.append(remaining[np.argmax(dists)])

    return descs[selected]


def aggregate_descriptors(map_points, normalize=True, max_k=4):
    all_descs = []
    all_xyzs = []

    for pid, mp in map_points.items():
        diverse = select_diverse_descriptors(mp.descs, max_k=max_k)

        for d in diverse:
            if normalize:
                d = d / (np.linalg.norm(d) + 1e-8)
            all_descs.append(d.astype(np.float32))
            all_xyzs.append(mp.xyz.astype(np.float32))

    return (
        np.vstack(all_descs),
        np.vstack(all_xyzs),
    )


# =============================
# -------- MAIN ---------------
# =============================

def main():
    parser = argparse.ArgumentParser(description="Build 3D map with optional pruning")

    parser.add_argument("--sparse_path", required=True)
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--output_path", required=True)

    parser.add_argument("--min_views", type=int, default=3)
    parser.add_argument("--max_descs_per_point", type=int, default=4)
    parser.add_argument("--no_normalize", action="store_true")

    # NEW
    parser.add_argument("--pruned_ply", default=None)
    parser.add_argument("--radius", type=float, default=0.01)

    args = parser.parse_args()

    points_path = os.path.join(args.sparse_path, "points3D.bin")

    print("Loading points3D...")
    points3D = read_points3D_binary(points_path)

    print(f"Total COLMAP points: {len(points3D)}")

    print("Loading descriptors...")
    descriptors = load_descriptors(args.database_path)

    tree = None
    if args.pruned_ply is not None:
        print("Loading pruned PLY...")
        pruned_xyz = load_ply_xyz(args.pruned_ply)
        print(f"Pruned PLY points: {len(pruned_xyz)}")

        tree = build_spatial_filter(pruned_xyz)

    print("Building map...")
    map_points = build_map(
        points3D,
        descriptors,
        args.min_views,
        tree=tree,
        radius=args.radius
    )

    print(f"Points after filtering: {len(map_points)}")

    print("Aggregating descriptors...")
    descs, xyzs = aggregate_descriptors(
        map_points,
        normalize=not args.no_normalize,
        max_k=args.max_descs_per_point,
    )

    print("Saving...")
    np.savez_compressed(args.output_path, xyzs=xyzs, descs=descs)

    print(f"Saved to {args.output_path}")
    print(f"xyzs: {xyzs.shape}, descs: {descs.shape}")


if __name__ == "__main__":
    main()
