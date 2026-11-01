import os
import shutil
import random
import glob

raw_dir = "data/raw_images"
dev_img_dir = "data/dev/images"
eval_img_dir = "data/eval/images"

os.makedirs(dev_img_dir, exist_ok=True)
os.makedirs(eval_img_dir, exist_ok=True)

for f in glob.glob(os.path.join(dev_img_dir, "*")):
    try:
        os.remove(f)
    except OSError:
        pass

for f in glob.glob(os.path.join(eval_img_dir, "*")):
    try:
        os.remove(f)
    except OSError:
        pass

extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
all_files = []
for ext in extensions:
    all_files.extend(glob.glob(os.path.join(raw_dir, "**", ext), recursive=True))

all_files = [f for f in all_files if not os.path.basename(f).startswith(".")]
all_files = sorted(list(set(all_files)))

total_images = len(all_files)
eval_target = int(round(total_images * 0.20))

block_size = 5
blocks = [all_files[i:i + block_size] for i in range(0, total_images, block_size)]

random.seed(42)
random.shuffle(blocks)

dev_files = []
eval_files = []

for block in blocks:
    needed = eval_target - len(eval_files)
    if needed <= 0:
        dev_files.extend(block)
    elif len(block) <= needed:
        eval_files.extend(block)
    else:
        eval_files.extend(block[:needed])
        dev_files.extend(block[needed:])

for src in dev_files:
    shutil.copy(src, os.path.join(dev_img_dir, os.path.basename(src)))

for src in eval_files:
    shutil.copy(src, os.path.join(eval_img_dir, os.path.basename(src)))

print(f"Total images found: {total_images}")
print(f"Dev split: {len(dev_files)} images ({(len(dev_files)/total_images)*100:.1f}%)")
print(f"Eval split: {len(eval_files)} images ({(len(eval_files)/total_images)*100:.1f}%)")