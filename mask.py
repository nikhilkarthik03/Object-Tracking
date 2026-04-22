import os
import cv2
import numpy as np
import argparse
import torch
from segment_anything import sam_model_registry, SamPredictor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", default="vit_b")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    print(f"Loading SAM on {device}...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    image_files = sorted(os.listdir(args.input_dir))

    for fname in image_files:
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        print(f"Processing: {fname}")

        img_path = os.path.join(args.input_dir, fname)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Skipping unreadable: {fname}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        h, w = image.shape[:2]

        # -----------------------------
        # BOX (tune if needed)
        # -----------------------------
        box = np.array([
            int(w * 0.35),
            int(h * 0.15),
            int(w * 0.65),
            int(h * 0.95),
        ])

        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True
        )

        if len(scores) == 0:
            print(f"No mask found: {fname}")
            continue

        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        # -----------------------------
        # Convert to proper COLMAP mask
        # -----------------------------
        mask = (mask > 0).astype(np.uint8) * 255

        # Smooth + stabilize (important for video)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, 1)
        mask = cv2.medianBlur(mask, 5)

        # -----------------------------
        # IMPORTANT: filename fix
        # -----------------------------
        out_name = fname + ".png"   # <-- key fix
        out_path = os.path.join(args.output_dir, out_name)

        cv2.imwrite(out_path, mask)

    print("Done.")


if __name__ == "__main__":
    main()