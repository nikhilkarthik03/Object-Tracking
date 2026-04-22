import open3d as o3d
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input PLY file")
    parser.add_argument("--output", default="cleaned.ply", help="Output PLY file")

    # Downsample
    parser.add_argument("--voxel_size", type=float, default=0.005)

    # Outlier removal
    parser.add_argument("--nb_neighbors", type=int, default=20)
    parser.add_argument("--std_ratio", type=float, default=2.0)

    # Plane removal
    parser.add_argument("--plane_dist", type=float, default=0.01)

    # Clustering
    parser.add_argument("--cluster_eps", type=float, default=0.02)
    parser.add_argument("--min_points", type=int, default=50)

    # Optional bounding box crop
    parser.add_argument("--use_bbox", action="store_true")
    parser.add_argument("--bbox", nargs=6, type=float,
                        help="minx miny minz maxx maxy maxz")

    args = parser.parse_args()

    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(args.input)
    print(f"Original points: {len(pcd.points)}")

    # -----------------------------
    # 1. Downsample
    # -----------------------------
    print("Downsampling...")
    pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
    print(f"After downsample: {len(pcd.points)}")

    # -----------------------------
    # 2. Remove outliers
    # -----------------------------
    print("Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio
    )
    print(f"After outlier removal: {len(pcd.points)}")

    # -----------------------------
    # 3. Remove dominant plane
    # -----------------------------
    print("Removing dominant plane...")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=args.plane_dist,
        ransac_n=3,
        num_iterations=1000
    )

    pcd = pcd.select_by_index(inliers, invert=True)
    print(f"After plane removal: {len(pcd.points)}")

    # -----------------------------
    # 4. Clustering (DBSCAN)
    # -----------------------------
    print("Clustering...")
    labels = np.array(
        pcd.cluster_dbscan(
            eps=args.cluster_eps,
            min_points=args.min_points,
            print_progress=True
        )
    )

    if labels.max() < 0:
        print("No clusters found, saving current result.")
        o3d.io.write_point_cloud(args.output, pcd)
        return

    # Keep largest cluster
    print("Selecting largest cluster...")
    largest_cluster = max(set(labels), key=lambda x: np.sum(labels == x))
    indices = np.where(labels == largest_cluster)[0]
    pcd = pcd.select_by_index(indices)

    print(f"After clustering: {len(pcd.points)}")

    # -----------------------------
    # 5. Optional bounding box crop
    # -----------------------------
    if args.use_bbox and args.bbox is not None:
        print("Applying bounding box crop...")
        min_bound = args.bbox[:3]
        max_bound = args.bbox[3:]

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound,
            max_bound=max_bound
        )

        pcd = pcd.crop(bbox)
        print(f"After bbox crop: {len(pcd.points)}")

    # -----------------------------
    # 6. Save result
    # -----------------------------
    o3d.io.write_point_cloud(args.output, pcd)
    print(f"Saved cleaned point cloud to: {args.output}")

    # -----------------------------
    # Optional visualization
    # -----------------------------
    print("Visualizing result...")
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()