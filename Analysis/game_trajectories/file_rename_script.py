import os
import re
import shutil # For potential future use like copying, though not used for renaming

def renumber_reversed_agent_runs(directory_path, dry_run=True):
    """
    Scans a directory for JSON files from agent comparisons.
    If it finds pairs like 'AgentA vs AgentB (1..20).json' and 'AgentB vs AgentA (1..20).json',
    it renames the 'AgentB vs AgentA' files to 'AgentA vs AgentB (run_num + 20).json',
    assuming AgentA comes alphabetically before AgentB.

    Args:
        directory_path (str): The path to the directory containing the JSON files.
        dry_run (bool): If True, only print proposed changes. If False, perform renames.
    """

    # Corrected Regex:
    # The main change is in the (?P<gamelendetails>) group: removed the leading space from its pattern,
    # as the space separating agent2 from gamelendetails is already matched by the literal space
    # after the agent2 group.
    filename_pattern = re.compile(
        r"^(?P<agent1>.*?) vs (?P<agent2>.*?) (?P<gamelendetails>\d+ sec .*?)(?P<run_open_space> \() *(?P<run_digits>\d+)(?P<run_close>\)\.json)$"
    )
    # Example: "Agent X vs Agent Y 60 sec Layout (1).json"
    # (?P<agent1>.*?)                  -> "Agent X"
    #  vs                              -> " vs "
    # (?P<agent2>.*?)                  -> "Agent Y"
    #                                  -> literal space (matches space after "Agent Y")
    # (?P<gamelendetails>\d+ sec .*?)  -> "60 sec Layout" (captures game length, "sec", and layout name)
    # (?P<run_open_space> \()          -> " (" (space before parenthesis and the parenthesis itself)
    #  * -> zero or more spaces (e.g. if filename was "( 1).json")
    # (?P<run_digits>\d+)              -> "1"
    # (?P<run_close>\)\.json)$         -> ").json"


    renamed_count = 0
    skipped_due_to_conflict = 0
    skipped_due_to_pattern_mismatch = 0
    processed_files_count = 0
    files_to_rename = []

    print(f"--- Starting Agent Run Renumbering ---")
    if dry_run:
        print("DRY RUN MODE: No files will actually be renamed.")
    else:
        print("EXECUTION MODE: Files will be renamed.")
    print(f"Using Regex: {filename_pattern.pattern}") # Print the regex for debugging
    print(f"Scanning directory: {directory_path}\n")

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            processed_files_count += 1
            original_filepath = os.path.join(directory_path, filename)
            match = filename_pattern.match(filename)

            if match:
                data = match.groupdict()
                agent1_orig = data['agent1']
                agent2_orig = data['agent2']
                
                # Determine canonical order
                sorted_agents = sorted([agent1_orig, agent2_orig])
                canonical_agent1 = sorted_agents[0]
                canonical_agent2 = sorted_agents[1]

                # Check if the original filename was in non-canonical order
                if agent1_orig != canonical_agent1: # e.g., original was "AgentB vs AgentA"
                    try:
                        run_num_int_orig = int(data['run_digits'])
                    except ValueError:
                        print(f"  Skipping (could not parse run number as int from '{data['run_digits']}'): {filename}")
                        skipped_due_to_pattern_mismatch +=1 
                        continue

                    new_run_num_int = run_num_int_orig + 20
                    
                    # Reconstruct the new filename
                    # data['gamelendetails'] is "60 sec Cramped Room"
                    # data['run_open_space'] is " ("
                    # data['run_digits'] is the original run number string
                    # data['run_close'] is ").json"
                    
                    new_filename = (
                        f"{canonical_agent1} vs {canonical_agent2}"  # "AgentA vs AgentB"
                        f" {data['gamelendetails']}"                 # " 60 sec Cramped Room" (adds leading space)
                        f"{data['run_open_space']}"                 # " ("
                        f"{str(new_run_num_int)}"                   # new run number
                        f"{data['run_close']}"                      # ").json"
                    )
                    new_filepath = os.path.join(directory_path, new_filename)

                    files_to_rename.append({
                        "original_filename": filename,
                        "original_filepath": original_filepath,
                        "new_filename": new_filename,
                        "new_filepath": new_filepath,
                        "reason": f"Reversed agent order ('{agent1_orig}' vs '{agent2_orig}'); "
                                  f"run {run_num_int_orig} -> {new_run_num_int}"
                    })
                # else: file is already in canonical order, or agents are the same (A vs A)
            else:
                print(f"  Skipping (filename does not match expected pattern): {filename}")
                skipped_due_to_pattern_mismatch += 1
    
    if not files_to_rename:
        print("No files found that require renumbering based on reversed agent order.")
    else:
        print("\n--- Proposed Renames ---")
        for item in files_to_rename:
            print(f"  Rename: '{item['original_filename']}'")
            print(f"      TO: '{item['new_filename']}'")
            print(f"  Reason: {item['reason']}")

        if not dry_run:
            print("\n--- Performing Renames ---")
            for item in files_to_rename:
                original_filepath = item['original_filepath']
                new_filepath = item['new_filepath']
                
                if os.path.exists(new_filepath):
                    print(f"  CONFLICT: Target file '{item['new_filename']}' already exists. Skipping rename of '{item['original_filename']}'.")
                    skipped_due_to_conflict += 1
                else:
                    try:
                        os.rename(original_filepath, new_filepath)
                        print(f"  SUCCESS: Renamed '{item['original_filename']}' to '{item['new_filename']}'")
                        renamed_count += 1
                    except Exception as e:
                        print(f"  ERROR: Failed to rename '{item['original_filename']}'. Reason: {e}")
                        skipped_due_to_conflict +=1 # Or a new error counter

    print(f"\n--- Renumbering Complete ---")
    print(f"  Total JSON files processed: {processed_files_count}")
    print(f"  Files renamed: {renamed_count}")
    print(f"  Files skipped due to naming pattern mismatch: {skipped_due_to_pattern_mismatch}")
    print(f"  Files skipped due to target name conflict (or error during rename): {skipped_due_to_conflict}")
    if dry_run and files_to_rename:
        print("\nReminder: This was a DRY RUN. No files were actually changed.")
        print("To apply changes, run the script with the dry_run parameter set to False.")

# --- How to use this script ---
if __name__ == "__main__":
    # --- Configuration ---
    PERFORM_DRY_RUN = False # ALWAYS RUN WITH dry_run=True FIRST TO VERIFY!

    # === OPTION 1: Specify an ABSOLUTE path to your JSON files ===
    # If you use this, the 'relative_path_parts' below will be ignored.
    # Uncomment the next line and set your path. Use r"..." for Windows paths.
    # directory_to_process = r"D:\Projects\AI\overcooked_ai\Analysis\game_trajectories\Cramped Room"
    # --- End Option 1 ---


    # === OPTION 2: Use a path RELATIVE to this script's location ===
    # Adjust 'relative_path_parts' based on where this script is saved.
    #
    # Example 1: Script is in 'D:\Projects\AI\overcooked_ai\Analysis\'
    #            Target is 'D:\Projects\AI\overcooked_ai\Analysis\game_trajectories\Cramped Room\'
    #            Then: relative_path_parts = ["game_trajectories", "Cramped Room"]
    #
    # Example 2: Script is in 'D:\Projects\AI\overcooked_ai\Analysis\game_trajectories\' (YOUR SITUATION)
    #            Target is 'D:\Projects\AI\overcooked_ai\Analysis\game_trajectories\Cramped Room\'
    #            Then: relative_path_parts = ["Cramped Room"] 
    #
    # Example 3: Script is in the SAME FOLDER as the JSON files (e.g., in 'Cramped Room\')
    #            Then: relative_path_parts = [] (empty list)

    relative_path_parts = ["Cramped Room"] # ADJUST THIS BASED ON YOUR SCRIPT LOCATION
    # --- End Option 2 ---

    # --- Path Setup Logic (Do not change from here unless you know what you're doing) ---
    user_defined_absolute_path = None
    # Check if 'directory_to_process' was defined and set to an absolute path by the user
    if 'directory_to_process' in locals() and isinstance(locals()['directory_to_process'], str):
        if os.path.isabs(locals()['directory_to_process']):
             user_defined_absolute_path = locals()['directory_to_process']

    if user_defined_absolute_path:
        final_directory_to_process = user_defined_absolute_path
        print(f"Using user-defined absolute path: {final_directory_to_process}")
    else:
        try:
            script_location_dir = os.path.dirname(os.path.realpath(__file__))
        except NameError: 
            script_location_dir = os.getcwd()
            print(f"Warning: __file__ not defined. Using current working directory as script's base: {script_location_dir}")
        
        if relative_path_parts is not None: # Can be an empty list
             final_directory_to_process = os.path.join(script_location_dir, *relative_path_parts)
        else: # Should not happen if relative_path_parts is defined, but as a fallback
            final_directory_to_process = script_location_dir 
        print(f"Script base location: {script_location_dir}")
        print(f"Using relative path. Target directory for processing: {final_directory_to_process}")
    # --- End Path Setup Logic ---

    if not os.path.isdir(final_directory_to_process):
        print(f"\nError: Target directory '{final_directory_to_process}' is not valid or not configured.")
        print("Please check your path configuration at the bottom of the script:")
        print("1. If using ABSOLUTE path, ensure 'directory_to_process' is uncommented and correct (use r\"...\" for Windows).")
        print("2. If using RELATIVE path, ensure 'relative_path_parts' is set correctly based on where you saved this script.")
    else:
        renumber_reversed_agent_runs(final_directory_to_process, dry_run=PERFORM_DRY_RUN)
