from pathlib import Path
from PIL import Image
import shutil
import sys


def resize_pngs(
    src_dir="../report/figs",
    dst_dir=None,
    max_width=1200,
):
    """
    Resize all .png images in src_dir to have width <= max_width,
    preserving aspect ratio. Writes to dst_dir. Originals are not modified.
    """
    src = Path(src_dir)
    dst = Path(dst_dir) if dst_dir else src.with_name(src.name + f"_{max_width}w")
    dst.mkdir(parents=True, exist_ok=True)

    resized = 0
    kept = 0

    for p in src.glob("*.png"):
        with Image.open(p) as im:
            w, h = im.size

            # skip upscaling
            if w <= max_width:
                shutil.copy2(p, dst / p.name)
                kept += 1
                continue

            new_w = max_width
            new_h = round(h * new_w / w)

            # palette images -> convert to avoid odd artifacts on resize
            if im.mode == "P":
                im = im.convert("RGBA")

            im_resized = im.resize((new_w, new_h), Image.LANCZOS)
            im_resized.save(dst / p.name, optimize=True)
            resized += 1

    print(f"Done. Resized: {resized}, kept as-is: {kept}. Output: {dst}")


if __name__ == "__main__":
    # Usage:
    #   python resize_pngs.py [src_dir] [dst_dir] [max_width]
    # Examples:
    #   python resize_pngs.py
    #   python resize_pngs.py ../report/figs ../report/figs_1200w 1200
    src_arg = sys.argv[1] if len(sys.argv) > 1 else "../report/figs"
    dst_arg = sys.argv[2] if len(sys.argv) > 2 else None
    width_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
    resize_pngs(src_arg, dst_arg, width_arg)