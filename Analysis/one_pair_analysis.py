import os
import json
import math
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import re

ALL_AGENT_NAMES = sorted([
    "Human-aware PPO",
    "Population-Based Training",
    "Self-Play",
    "Human Keyboard Input"
])


def compute_entropy(action_list):
    if not action_list:
        return 0.0
    counts = Counter(str(a) for a in action_list)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total, 2) for c in counts.values() if c > 0)


def detect_coordination_conflicts(observations, actions):
    conflict_count = 0
    max_len = min(len(observations), len(actions))
    for t in range(1, max_len):
        if len(observations[t].get("players", [])) < 2 or \
                observations[t]["players"][0] is None or \
                observations[t]["players"][1] is None or \
                observations[t]["players"][0].get("position") is None or \
                observations[t]["players"][1].get("position") is None:
            continue
        pos0 = tuple(observations[t]["players"][0]["position"])
        pos1 = tuple(observations[t]["players"][1]["position"])
        if t >= len(actions) or len(actions[t]) < 2:
            continue
        act0 = actions[t][0]
        act1 = actions[t][1]
        both_interact = (act0 == "INTERACT" or (isinstance(act0, list) and "INTERACT" in act0)) and \
                        (act1 == "INTERACT" or (isinstance(
                            act1, list) and "INTERACT" in act1))
        near_same_tile = abs(pos0[0] - pos1[0]) + abs(pos0[1] - pos1[1]) <= 1
        if both_interact and near_same_tile:
            conflict_count += 1
    return conflict_count


def parse_filename_for_agents_and_episode(filename):
    match = re.match(
        r"^(.*) vs (.*) \d+ sec Cramped Room \((\d+)\)\.json$", filename)
    if not match:
        print(f"[DEBUG] Filename did not match regex: {filename}")
        return None

    raw_agent1 = match.group(1).strip()
    raw_agent2 = match.group(2).strip()
    episode_number = int(match.group(3))

    # Normalize whitespace
    normalized_agent1 = ' '.join(raw_agent1.split())
    normalized_agent2 = ' '.join(raw_agent2.split())

    # All known agent names
    valid_agents = {
        name.lower(): name for name in [
            "Human Keyboard Input",
            "Human-aware PPO",
            "Population-Based Training",
            "Self-Play"
        ]
    }

    if normalized_agent1.lower() in valid_agents and normalized_agent2.lower() in valid_agents:
        return (
            valid_agents[normalized_agent1.lower()],
            valid_agents[normalized_agent2.lower()],
            episode_number
        )
    else:
        print(
            f"[DEBUG] Unrecognized agent pair: '{normalized_agent1}' vs '{normalized_agent2}' in file: {filename}")
        return None


def get_stats_and_distance_series_from_json_file(filepath, agent1_name_from_filename, agent2_name_from_filename):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception:
        return None

    if not data.get('ep_observations') or not data['ep_observations'][0]:
        return None
    observations = data['ep_observations'][0]
    episode_length = len(observations)
    if episode_length == 0:
        return None

    actions = data.get('ep_actions', [[]])[0] if data.get(
        'ep_actions') else [[] for _ in range(episode_length)]
    rewards = data.get('ep_rewards', [[]])[0] if data.get(
        'ep_rewards') else [0 for _ in range(episode_length)]

    if len(actions) < episode_length:
        actions.extend([[] for _ in range(episode_length - len(actions))])
    if len(rewards) < episode_length:
        rewards.extend([0 for _ in range(episode_length - len(rewards))])

    if not (observations[0].get("players") and len(observations[0].get("players")) == 2):
        return None

    distance_series = [None] * episode_length
    interact_counts = [0, 0]
    idle_counts = [0, 0]
    action_sequences = [[], []]

    for t in range(episode_length):
        current_players_in_obs = observations[t].get("players", [])
        if len(current_players_in_obs) == 2:
            p0_data = current_players_in_obs[0]
            p1_data = current_players_in_obs[1]
            if p0_data and p0_data.get("position") and p1_data and p1_data.get("position"):
                pos0, pos1 = tuple(p0_data["position"]), tuple(
                    p1_data["position"])
                distance_series[t] = math.sqrt(
                    (pos0[0] - pos1[0]) ** 2 + (pos0[1] - pos1[1]) ** 2)

            if t < len(actions) and len(actions[t]) == 2:
                for i in [0, 1]:
                    act = actions[t][i]
                    action_sequences[i].append(act)
                    if act == "INTERACT" or (isinstance(act, list) and "INTERACT" in act):
                        interact_counts[i] += 1
                    # Handle list or tuple for idle
                    elif act == [0, 0] or act == (0, 0):
                        idle_counts[i] += 1

    total_reward = sum(rewards)
    soups_delivered = rewards.count(20)
    reward_efficiency = total_reward / episode_length if episode_length > 0 else 0
    entropies = [compute_entropy(seq) for seq in action_sequences]
    conflict_count = detect_coordination_conflicts(observations, actions)

    division_of_labor = [0, 0]
    reaction_times_list = []
    reward_events = [(i, r) for i, r in enumerate(rewards) if r > 0]
    credited_rewards_for_dol = set()

    for t_reward, r_value in reward_events:
        if r_value == 20 and t_reward not in credited_rewards_for_dol:
            found_deliverer_for_event = False
            for dt in range(3):
                check_t = t_reward - dt
                if not (0 <= check_t < len(observations)):
                    continue
                obs_players_at_check_t = observations[check_t].get(
                    "players", [])
                if len(obs_players_at_check_t) < 2:
                    continue
                for agent_idx_in_json in [0, 1]:
                    player_data = obs_players_at_check_t[agent_idx_in_json]
                    if player_data and player_data.get("held_object", {}).get("name") == "soup":
                        division_of_labor[agent_idx_in_json] += 1
                        reaction_times_list.append(dt)
                        credited_rewards_for_dol.add(t_reward)
                        found_deliverer_for_event = True
                        break
                if found_deliverer_for_event:
                    break

    avg_reaction_time = sum(reaction_times_list) / \
        len(reaction_times_list) if reaction_times_list else None

    return {
        "distance_series": distance_series,
        "agent1_name_in_file": agent1_name_from_filename,
        "agent2_name_in_file": agent2_name_from_filename,
        "total_reward": total_reward,
        "soups_delivered": soups_delivered,
        "reward_efficiency": reward_efficiency,
        "conflict_count": conflict_count,
        "avg_reaction_time_pair": avg_reaction_time,
        "interact_p0": interact_counts[0], "idle_p0": idle_counts[0],
        "entropy_p0": entropies[0], "deliveries_p0": division_of_labor[0],
        "interact_p1": interact_counts[1], "idle_p1": idle_counts[1],
        "entropy_p1": entropies[1], "deliveries_p1": division_of_labor[1],
        "ep_rewards": rewards,
    }


def analyze_and_plot_average_stats_and_distances(folder_path, episodes_to_average=range(1, 21)):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"The specified folder does not exist: {folder_path}")

    all_filenames_in_folder = os.listdir(folder_path)
    if not all_filenames_in_folder:
        print(f"No files found in folder: {folder_path}")
        return

    print(
        f"Processing files in '{folder_path}' for episodes {min(episodes_to_average)}-{max(episodes_to_average)}...")
    aggregated_pair_episode_data = defaultdict(list)
    max_overall_timesteps = 0
    processed_file_count = 0

    for filename in all_filenames_in_folder:
        if not filename.endswith(".json"):
            continue
        parsed_info = parse_filename_for_agents_and_episode(filename)
        if parsed_info:
            agent1_name, agent2_name, episode_number = parsed_info
            if episode_number in episodes_to_average:
                processed_file_count += 1
                filepath = os.path.join(folder_path, filename)
                stats_and_series = get_stats_and_distance_series_from_json_file(
                    filepath, agent1_name, agent2_name)
                if stats_and_series:
                    canonical_pair_key = tuple(
                        sorted((agent1_name, agent2_name)))
                    aggregated_pair_episode_data[canonical_pair_key].append(
                        stats_and_series)
                    max_overall_timesteps = max(max_overall_timesteps, len(
                        stats_and_series["distance_series"]))
        else:
            print(f"Skipped unrecognized file: {filename}")

    if processed_file_count == 0:
        print(f"No files matching criteria were processed.")
        return
    if not aggregated_pair_episode_data:
        print("No data successfully aggregated. Cannot generate reports or plots.")
        return

    print(
        f"\n--- Averaged Statistics, Distance Plots & Histograms (Max Timesteps: {max_overall_timesteps}) ---")

    for canonical_pair, episode_data_list in aggregated_pair_episode_data.items():
        N1_name, N2_name = canonical_pair
        num_episodes_for_this_pair = len(episode_data_list)
        print(
            f"\n=== Pair: {N1_name} vs {N2_name} (based on {num_episodes_for_this_pair} episodes) ===")

        pair_level_stats_aggr = defaultdict(list)
        N1_stats_aggr = defaultdict(list)
        N2_stats_aggr = defaultdict(list)
        all_distance_series_for_pair = []
        all_individual_distances_for_pair = []  # For histogram

        for ep_data in episode_data_list:
            all_distance_series_for_pair.append(ep_data["distance_series"])
            for dist_val in ep_data["distance_series"]:
                if dist_val is not None and not np.isnan(dist_val):
                    all_individual_distances_for_pair.append(dist_val)

            pair_level_stats_aggr["total_reward"].append(
                ep_data["total_reward"])
            pair_level_stats_aggr["soups_delivered"].append(
                ep_data["soups_delivered"])
            pair_level_stats_aggr["reward_efficiency"].append(
                ep_data["reward_efficiency"])
            pair_level_stats_aggr["conflict_count"].append(
                ep_data["conflict_count"])
            if ep_data["avg_reaction_time_pair"] is not None:
                pair_level_stats_aggr["avg_reaction_time_pair"].append(
                    ep_data["avg_reaction_time_pair"])

            current_file_agent1 = ep_data["agent1_name_in_file"]
            if current_file_agent1 == N1_name:
                N1_stats_aggr["interact"].append(ep_data["interact_p0"])
                N1_stats_aggr["idle"].append(ep_data["idle_p0"])
                N1_stats_aggr["entropy"].append(ep_data["entropy_p0"])
                N1_stats_aggr["deliveries"].append(ep_data["deliveries_p0"])
                N2_stats_aggr["interact"].append(ep_data["interact_p1"])
                N2_stats_aggr["idle"].append(ep_data["idle_p1"])
                N2_stats_aggr["entropy"].append(ep_data["entropy_p1"])
                N2_stats_aggr["deliveries"].append(ep_data["deliveries_p1"])
            else:
                N2_stats_aggr["interact"].append(ep_data["interact_p0"])
                N2_stats_aggr["idle"].append(ep_data["idle_p0"])
                N2_stats_aggr["entropy"].append(ep_data["entropy_p0"])
                N2_stats_aggr["deliveries"].append(ep_data["deliveries_p0"])
                N1_stats_aggr["interact"].append(ep_data["interact_p1"])
                N1_stats_aggr["idle"].append(ep_data["idle_p1"])
                N1_stats_aggr["entropy"].append(ep_data["entropy_p1"])
                N1_stats_aggr["deliveries"].append(ep_data["deliveries_p1"])

        print("  Pair-Level Average Stats:")
        for stat_name, values in pair_level_stats_aggr.items():
            avg_val = sum(values) / len(values) if values else 0
            print(
                f"    Avg. {stat_name.replace('_', ' ').title()}: {avg_val:.2f}")

        for agent_name_in_pair, agent_stats_aggr in [(N1_name, N1_stats_aggr), (N2_name, N2_stats_aggr)]:
            print(f"  Average Stats for {agent_name_in_pair} (in this pair):")
            for stat_name, values in agent_stats_aggr.items():
                avg_val = sum(values) / len(values) if values else 0
                print(
                    f"    Avg. {stat_name.replace('_', ' ').title()}: {avg_val:.2f}")

        padded_series_list = []
        for s in all_distance_series_for_pair:
            new_s = [np.nan] * max_overall_timesteps
            for i in range(min(len(s), max_overall_timesteps)):
                if s[i] is not None:
                    new_s[i] = s[i]
            padded_series_list.append(new_s)

        if not padded_series_list:
            print(
                f"  No distance series data to plot average for pair: {N1_name} vs {N2_name}.")
        else:
            try:
                averaged_distances = np.nanmean(
                    np.array(padded_series_list), axis=0)
            except Exception:
                averaged_distances = np.array([np.nan] * max_overall_timesteps)

            plot_timesteps = np.arange(max_overall_timesteps)
            valid_indices = ~np.isnan(averaged_distances)
            plot_timesteps_valid, plot_values_valid = plot_timesteps[
                valid_indices], averaged_distances[valid_indices]

            if len(plot_values_valid) > 0:
                plt.figure(figsize=(13, 7))
                plt.plot(plot_timesteps_valid, plot_values_valid, marker='.', linestyle='-', markersize=4,
                         label=f"Avg. Distance")
                plt.title(
                    f"Euclidean Distance Over Time\n{N1_name} vs {N2_name} (Avg over {num_episodes_for_this_pair} episodes)",
                    fontsize=14
                )

                plt.xlabel("Timestep", fontsize=12)
                plt.ylabel("Average Euclidean Distance", fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                plt.tight_layout()
                print(
                    f"  Displaying average distance plot for: {N1_name} vs {N2_name}")
                plt.show()
            else:
                print(
                    f"  No valid averaged distance data to plot for pair: {N1_name} vs {N2_name}.")

        # Assuming averaged_distances is a NumPy array
        # Remove NaN values before plotting
        valid_distances = averaged_distances[~np.isnan(averaged_distances)]

        # Create histogram
        plt.figure(figsize=(8, 5))
        plt.hist(valid_distances, bins=30, edgecolor='black')
        plt.title(
            f"Histogram of Averaged Distances\n{N1_name} vs {N2_name}", fontsize=13)
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def compute_and_plot_95th_percentiles(aggregated_pair_episode_data):
    """
    Computes the 95th percentile of Euclidean distances for each agent pair across episodes,
    sorts them, and plots a horizontal bar chart.
    """
    print("\n--- 95th Percentile Euclidean Distance by Agent Pair ---")
    pair_95th_percentiles = []

    for canonical_pair, episode_data_list in aggregated_pair_episode_data.items():
        all_distances = []
        for ep_data in episode_data_list:
            all_distances.extend([d for d in ep_data["distance_series"] if d is not None and not np.isnan(d)])
        if all_distances:
            percentile_95 = np.percentile(all_distances, 95)
            pair_95th_percentiles.append((canonical_pair, percentile_95))
            print(f"{canonical_pair}: 95th Percentile = {percentile_95:.2f}")
        else:
            print(f"{canonical_pair}: No valid distance data.")

    # Sort pairs by 95th percentile value descending
    pair_95th_percentiles.sort(key=lambda x: x[1], reverse=True)

    # Prepare data for plotting
    labels = [f"{p[0][0]} vs {p[0][1]}" for p in pair_95th_percentiles]
    values = [p[1] for p in pair_95th_percentiles]

    # Plot if data exists
    if values:
        plt.figure(figsize=(12, 6))
        plt.barh(labels, values, edgecolor='black')
        plt.xlabel("95th Percentile of Euclidean Distance", fontsize=12)
        plt.title("Sorted 95th Percentiles of Euclidean Distance by Agent Pair", fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid distance data to plot 95th percentile chart.")

def compute_aggregated_pair_episode_data(folder_path, episodes_to_average=range(1, 21)):
    """
    Scans the specified folder, parses Overcooked-AI JSON logs,
    and returns a dictionary of aggregated episode stats by agent pair.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The specified folder does not exist: {folder_path}")

    all_filenames_in_folder = os.listdir(folder_path)
    if not all_filenames_in_folder:
        print(f"No files found in folder: {folder_path}")
        return {}

    aggregated_pair_episode_data = defaultdict(list)
    processed_file_count = 0

    for filename in all_filenames_in_folder:
        if not filename.endswith(".json"):
            continue
        parsed_info = parse_filename_for_agents_and_episode(filename)
        if parsed_info:
            agent1_name, agent2_name, episode_number = parsed_info
            if episode_number in episodes_to_average:
                processed_file_count += 1
                filepath = os.path.join(folder_path, filename)
                stats_and_series = get_stats_and_distance_series_from_json_file(
                    filepath, agent1_name, agent2_name)
                if stats_and_series:
                    canonical_pair_key = tuple(sorted((agent1_name, agent2_name)))
                    aggregated_pair_episode_data[canonical_pair_key].append(stats_and_series)
        else:
            print(f"Skipped unrecognized file: {filename}")

    if processed_file_count == 0:
        print(f"No files matching criteria were processed.")
        return {}

    return aggregated_pair_episode_data

def compute_and_plot_reward_95th_percentiles(aggregated_pair_episode_data):
    """
    Computes the 95th percentile of total episode rewards for each agent pair,
    plots reward histograms and a sorted bar chart of percentiles.
    """
    print("\n--- 95th Percentile of Total Episode Rewards by Agent Pair ---")
    pair_reward_95th_percentiles = []

    for canonical_pair, episode_data_list in aggregated_pair_episode_data.items():
        print(f"Analyzing {canonical_pair} with {len(episode_data_list)} episodes...")

        total_rewards = [
            ep["total_reward"]
            for ep in episode_data_list
            if "total_reward" in ep and ep["total_reward"] is not None
        ]

        print(f"  → Total rewards collected: {total_rewards}")

        if len(total_rewards) > 0:
            # Even if all are zeros, include the pair
            percentile_95 = np.percentile(total_rewards, 95)
            pair_reward_95th_percentiles.append((canonical_pair, percentile_95))
            print(f"  → 95th Percentile = {percentile_95:.2f}")

            # Histogram for this pair
            plt.figure(figsize=(8, 5))
            plt.hist(total_rewards, bins=15, edgecolor='black')
            plt.title(f"Histogram of Total Rewards\n{canonical_pair[0]} vs {canonical_pair[1]}", fontsize=13)
            plt.xlabel('Total Episode Reward')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"  → Skipped (no reward data)")

    # Sort for overall bar chart
    pair_reward_95th_percentiles.sort(key=lambda x: x[1], reverse=True)
    labels = [f"{p[0][0]} vs {p[0][1]}" for p in pair_reward_95th_percentiles]
    values = [p[1] for p in pair_reward_95th_percentiles]

    if values:
        plt.figure(figsize=(12, 6))
        bars = plt.barh(labels, values, edgecolor='black')
        for i, val in enumerate(values):
            plt.text(val + 1, i, f"{val:.0f}", va='center')  # Add value labels next to bars

        plt.xlabel("95th Percentile of Total Episode Reward", fontsize=12)
        plt.title("Sorted 95th Percentiles of Total Rewards by Agent Pair", fontsize=14)
        plt.gca().invert_yaxis()  # Highest at top
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid reward data to plot 95th percentile chart.")

def compute_and_plot_reward_intervals(aggregated_pair_episode_data, max_n_rewards_to_track=5):
    """
    Computes and plots the average time between successive rewards (e.g., 0→1, 1→2, ..., 4→5).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    print("\n--- Average Reward Intervals by Agent Pair ---")
    pair_reward_intervals = {}
    interval_index_to_timings = defaultdict(list)  # index = interval (0→1, 1→2, ...)

    for canonical_pair, episode_data_list in aggregated_pair_episode_data.items():
        intervals_by_index = [[] for _ in range(max_n_rewards_to_track)]

        for ep_data in episode_data_list:
            rewards = ep_data.get("ep_rewards", [])
            if not rewards:
                continue

            reward_timesteps = [i for i, r in enumerate(rewards) if r == 20]
            if not reward_timesteps:
                continue

            # Insert virtual step 0 for the first interval (0 → 1)
            reward_timesteps = [0] + reward_timesteps[:max_n_rewards_to_track]

            for i in range(1, len(reward_timesteps)):
                interval = reward_timesteps[i] - reward_timesteps[i - 1]
                intervals_by_index[i - 1].append(interval)

        avg_intervals = []
        for i, intervals in enumerate(intervals_by_index):
            if intervals:
                avg = sum(intervals) / len(intervals)
                avg_intervals.append(avg)
                interval_index_to_timings[i].append((canonical_pair, avg))
            else:
                avg_intervals.append(None)

        pair_reward_intervals[canonical_pair] = avg_intervals

    # --- Plot 1: Line chart per agent pair ---
    for canonical_pair, avg_intervals in pair_reward_intervals.items():
        interval_labels = [f"{i}→{i+1}" for i in range(max_n_rewards_to_track)]
        values = [v if v is not None else np.nan for v in avg_intervals]

        plt.figure(figsize=(8, 5))
        plt.plot(interval_labels, values, marker='o', label='Avg. Interval')
        plt.title(f"Avg. Interval Between Rewards\n{canonical_pair[0]} vs {canonical_pair[1]}", fontsize=13)
        plt.xlabel("Reward Interval")
        plt.ylabel("Average Time Between Rewards")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # --- Plot 2: Sorted bar charts for each interval index ---
    print("\n--- Sorted Avg. Interval Between Rewards Across Pairs ---")
    for idx in range(max_n_rewards_to_track):
        timing_list = interval_index_to_timings[idx]
        if timing_list:
            timing_list.sort(key=lambda x: x[1])
            labels = [f"{p[0]} vs {p[1]}" for p, _ in timing_list]
            values = [v for _, v in timing_list]

            print(f"\nInterval {idx}→{idx+1}:")
            for pair, avg_step in timing_list:
                print(f"  {pair}: Interval = {avg_step:.2f}")

            # Plot bar chart
            plt.figure(figsize=(10, 6))
            plt.barh(labels, values, edgecolor='black')
            for i, val in enumerate(values):
                plt.text(val + 2, i, f"{val:.1f}", va='center', fontsize=9)
            plt.xlabel("Average Time Between Rewards")
            plt.title(f"Avg. Interval {idx}→{idx+1} Between Rewards (Sorted)")
            plt.gca().invert_yaxis()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print(f"\nInterval {idx}→{idx+1}: No data")


folder_to_analyze = "./game_trajectories/Cramped Room"
episodes = range(1, 21)

if folder_to_analyze == "./game_trajectories/Cramped Room":
    print("--- Note: Using default example path for 'folder_to_analyze'. Ensure this is correct. ---")

print("\nAvailable Analyses:")
print("1. Average stats, distance plots, and histograms")
print("2. 95th Percentile of Euclidean Distances")
print("3. 95th Percentile of Total Rewards")
print("4. Average Timestep Between Each Reward (Reward Timing)")
print("5. Run All Analyses")

selection = input("\nEnter the number(s) of analysis to run (e.g., 1 or 1,3,4): ").strip()
selected_options = {int(s.strip()) for s in selection.split(',') if s.strip().isdigit()}

try:
    aggregated_data = compute_aggregated_pair_episode_data(folder_to_analyze, episodes_to_average=episodes)

    if 1 in selected_options:
        analyze_and_plot_average_stats_and_distances(
            folder_to_analyze, episodes_to_average=episodes)

    if any(i in selected_options for i in {2, 3, 4, 5}):
        aggregated_data = compute_aggregated_pair_episode_data(
            folder_to_analyze, episodes_to_average=episodes)

    if 2 in selected_options or 5 in selected_options:
        compute_and_plot_95th_percentiles(aggregated_data)

    if 3 in selected_options or 5 in selected_options:
        compute_and_plot_reward_95th_percentiles(aggregated_data)

    if 4 in selected_options or 5 in selected_options:
        compute_and_plot_reward_intervals(aggregated_data, max_n_rewards_to_track=5)

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")