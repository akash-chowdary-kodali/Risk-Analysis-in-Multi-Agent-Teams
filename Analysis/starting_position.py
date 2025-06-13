import os
import json
import re
from collections import defaultdict

def check_starting_positions(folder_path):
    """
    Analyzes all JSON game trajectories in a folder to determine if the
    agent starting positions are the same for every episode. This version uses
    the filename as a unique key to correctly count all files.
    """
    starting_positions = {}
    
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' was not found.")
        return

    try:
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        if not json_files:
            print(f"No JSON files found in '{folder_path}'")
            return
    except Exception as e:
        print(f"Error reading directory '{folder_path}': {e}")
        return

    for filename in json_files:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            first_observation = data.get('ep_observations', [[]])[0][0]
            players = first_observation.get('players', [])
            
            if len(players) >= 2 and players[0] and players[1]:
                pos0 = tuple(players[0]['position'])
                pos1 = tuple(players[1]['position'])
                start_config = tuple(sorted((pos0, pos1)))
                
                # Use the unique filename as the key
                starting_positions[filename] = start_config
            else:
                print(f"Warning: Could not find two players in the first observation of {filename}")

        except (IOError, json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"Could not process file {filename}: {e}")

    if not starting_positions:
        print("Could not extract any starting positions from the files.")
        return

    # --- Report the findings ---
    # Group files by their starting configuration
    config_to_files = defaultdict(list)
    for fname, config in starting_positions.items():
        config_to_files[config].append(fname)

    print("\n--- Analysis of Agent Starting Positions ---")
    # This count will now be correct
    print(f"Found and analyzed {len(starting_positions)} total trajectory files.")
    
    if len(config_to_files) == 1:
        first_config = next(iter(config_to_files.keys()))
        print("\n\033[1mResult: The starting positions for the agents are THE SAME for all trajectory files.\033[0m")
        print(f"The consistent starting configuration is: {first_config}")
    else:
        print("\n\033[1mResult: The starting positions for the agents are DIFFERENT across trajectory files.\033[0m")
        print("The following starting configurations were found:")
        for i, (config, files) in enumerate(config_to_files.items()):
            # To keep the output clean, show the config and how many files it appeared in
            print(f"  Configuration {i+1}: {config} (found in {len(files)} files)")

if __name__ == "__main__":
    # This path should point to the directory containing your JSON game files.
    FOLDER_TO_ANALYZE = ".\Analysis\game_trajectories\Cramped Room"
    
    check_starting_positions(FOLDER_TO_ANALYZE)