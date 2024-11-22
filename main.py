import argparse
from render import render_and_save_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_path", type=str, required=True, help="Path to the COLMAP sparse data directory.")
    parser.add_argument("--image_id", type=int, required=True, help="ID of the image to render.")
    args = parser.parse_args()

    render_and_save_image(colmap_path=args.colmap_path, image_idx=args.image_id)

if __name__ == "__main__":
    main()
