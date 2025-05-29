import os
import shutil
import re

source_folder = "Analysis/game_trajectories/Cramped Room"
target_folder = "Analysis/game_trajectories/CrampedRoom21to40"

# Ensure target folder exists
os.makedirs(target_folder, exist_ok=True)

# Regex to find episodes numbered 21-40
pattern = re.compile(r'\((2[1-9]|3[0-9]|40)\)\.json$')

for filename in os.listdir(source_folder):
    if pattern.search(filename):
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)
        shutil.move(source_path, target_path)
        print(f"Moved: {filename}")
