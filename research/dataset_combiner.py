import os
import shutil

from tqdm import tqdm

folders = ["blackflame", "soulsmith", "unsouled"]


def load_metadata(filename, prefix):
    with open(filename) as f:
        lines = f.readlines()
        lines = [prefix + line for line in lines]
        return lines


def copy_clips(folder, dest_folder, prefix):
    for filename in tqdm(os.listdir(folder)):
        source = os.path.join(folder, filename)
        dest = os.path.join(dest_folder, prefix + filename)
        shutil.copy(source, dest)


os.makedirs("combined")

# Combine metadata
metadata = (
    load_metadata("blackflame_metadata.csv", "blackflame_")
    + load_metadata("soulsmith_metadata.csv", "soulsmith_")
    + load_metadata("unsouled_metadata.csv", "unsouled_")
)

with open("combined.csv", "w", encoding="utf-8") as f:
    for line in metadata:
        f.write(f"{line.strip()}\n")

# Copy files
for folder in folders:
    copy_clips(folder, "combined", folder + "_")
