import os
import json
import math
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from dtw import dtw # Required for Dynamic Time Warping

# This is the provided code from one_pair_analysis.py.
# Helper functions from the original script are reused to ensure consistency.

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
        return None

    raw_agent1 = match.group(1).strip()
    raw_agent2 = match.group(2).strip()
    episode_number = int(match.group(3))

    valid_agents = {
        name.lower(): name for name in [
            "Human Keyboard Input", "Human-aware PPO",
            "Population-Based Training", "Self-Play"
        ]
    }

    normalized_agent1 = ' '.join(raw_agent1.split()).lower()
    normalized_agent2 = ' '.join(raw_agent2.split()).lower()

    if normalized_agent1 in valid_agents and normalized_agent2 in valid_agents:
        return (
            valid_agents[normalized_agent1],
            valid_agents[normalized_agent2],
            episode_number
        )
    return None

def get_stats_and_distance_series_from_json_file(filepath, agent1_name, agent2_name):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception:
        return None

    if not data.get('ep_observations') or not data['ep_observations'][0]:
        return None

    observations = data['ep_observations'][0]
    episode_length = len(observations)
    if episode_length < 2: return None

    actions = data.get('ep_actions', [[]])[0]
    rewards = data.get('ep_rewards', [[]])[0]

    distance_series, manhattan_distance_series, chebyshev_distance_series, minkowski_p3_series = [], [], [], []
    path0, path1 = [], []

    for t in range(episode_length):
        players = observations[t].get("players", [])
        if len(players) == 2 and players[0] and players[1]:
            pos0 = np.array(players[0]["position"])
            pos1 = np.array(players[1]["position"])
            path0.append(pos0)
            path1.append(pos1)

            distance_series.append(np.linalg.norm(pos0 - pos1))
            manhattan_distance_series.append(np.linalg.norm(pos0 - pos1, ord=1))
            chebyshev_distance_series.append(np.linalg.norm(pos0 - pos1, ord=np.inf))
            minkowski_p3_series.append(np.power(np.sum(np.power(np.abs(pos0 - pos1), 3)), 1/3))
        else:
            for series in [distance_series, manhattan_distance_series, chebyshev_distance_series, minkowski_p3_series]:
                series.append(None)

    dtw_distance = None
    if len(path0) > 1 and len(path1) > 1:
        try:
            dtw_result = dtw(np.array(path0), np.array(path1), keep_internals=False, distance_only=True)
            dtw_distance = dtw_result.distance
        except Exception:
            dtw_distance = None

    total_reward = sum(rewards)
    soups_delivered = rewards.count(20)
    reward_efficiency = total_reward / episode_length if episode_length > 0 else 0
    conflict_count = detect_coordination_conflicts(observations, actions)

    return {
        "distance_series": distance_series,
        "manhattan_distance_series": manhattan_distance_series,
        "chebyshev_distance_series": chebyshev_distance_series,
        "minkowski_p3_series": minkowski_p3_series,
        "dtw_distance": dtw_distance,
        "total_reward": total_reward,
        "soups_delivered": soups_delivered,
        "reward_efficiency": reward_efficiency,
        "conflict_count": conflict_count,
        "ep_rewards": rewards
    }

def compute_aggregated_pair_episode_data(folder_path, episodes_to_average):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    all_filenames = os.listdir(folder_path)
    aggregated_data = defaultdict(list)

    for filename in all_filenames:
        if not filename.endswith(".json"): continue

        parsed_info = parse_filename_for_agents_and_episode(filename)
        if parsed_info:
            agent1_name, agent2_name, episode = parsed_info
            if episode in episodes_to_average:
                filepath = os.path.join(folder_path, filename)
                stats = get_stats_and_distance_series_from_json_file(filepath, agent1_name, agent2_name)
                if stats:
                    key = tuple(sorted((agent1_name, agent2_name)))
                    aggregated_data[key].append(stats)

    return aggregated_data

def calculate_and_rank_metrics(aggregated_data):
    pair_metrics_raw = defaultdict(dict)

    METRICS_TO_RANK = {
        "Total Reward": True,
        "Soups Delivered": True,
        "Reward Efficiency": True,
        "Conflict Count": False,
        "Time to First Reward": False,
        "95th Percentile Euclidean Dist": False,
        "95th Percentile Manhattan Dist": False,
        "95th Percentile Chebyshev Dist": False,
        "95th Percentile Minkowski p=3 Dist": False,
        "DTW Path Distance": False,
    }

    for pair, episodes_data in aggregated_data.items():
        pair_metrics_raw[pair]["Total Reward"] = np.mean([e["total_reward"] for e in episodes_data])
        pair_metrics_raw[pair]["Soups Delivered"] = np.mean([e["soups_delivered"] for e in episodes_data])
        pair_metrics_raw[pair]["Reward Efficiency"] = np.mean([e["reward_efficiency"] for e in episodes_data])
        pair_metrics_raw[pair]["Conflict Count"] = np.mean([e["conflict_count"] for e in episodes_data])
        first_reward_times = [e["ep_rewards"].index(20) for e in episodes_data if 20 in e["ep_rewards"]]
        pair_metrics_raw[pair]["Time to First Reward"] = np.mean(first_reward_times) if first_reward_times else 400

        dist_series_keys = {
            "95th Percentile Euclidean Dist": "distance_series",
            "95th Percentile Manhattan Dist": "manhattan_distance_series",
            "95th Percentile Chebyshev Dist": "chebyshev_distance_series",
            "95th Percentile Minkowski p=3 Dist": "minkowski_p3_series",
        }
        for metric_name, series_key in dist_series_keys.items():
            all_distances = [d for e in episodes_data for d in e[series_key] if d is not None]
            pair_metrics_raw[pair][metric_name] = np.percentile(all_distances, 95) if all_distances else np.nan

        all_dtw_distances = [e["dtw_distance"] for e in episodes_data if e["dtw_distance"] is not None]
        pair_metrics_raw[pair]["DTW Path Distance"] = np.mean(all_dtw_distances) if all_dtw_distances else np.nan

    final_scores = []
    for pair in aggregated_data.keys():
        total_score = 0
        individual_scores = {}
        for metric, higher_is_better in METRICS_TO_RANK.items():
            values = [data[metric] for data in pair_metrics_raw.values() if data.get(metric) is not None and not np.isnan(data[metric])]
            if not values:
                norm_score = 0.5
            else:
                min_val, max_val = min(values), max(values)
                current_val = pair_metrics_raw[pair].get(metric)

                if current_val is None or np.isnan(current_val):
                    norm_score = 0
                elif max_val == min_val:
                    norm_score = 0.5
                else:
                    norm_score = (current_val - min_val) / (max_val - min_val)

                if not higher_is_better and not (current_val is None or np.isnan(current_val)):
                    norm_score = 1 - norm_score

            individual_scores[metric] = norm_score * 100
            total_score += individual_scores[metric]

        final_scores.append({
            "pair": f"{pair[0]} vs {pair[1]}",
            "total_score": total_score,
            "individual_scores": individual_scores
        })

    final_scores.sort(key=lambda x: x["total_score"], reverse=True)
    return final_scores, list(METRICS_TO_RANK.keys())

def plot_all_rankings_individually(ranking_data, metrics):
    """
    Displays a separate plot for each individual metric (but does not save them).
    """
    if not ranking_data:
        print("No ranking data to plot.")
        return

    for metric_name in metrics:
        plt.figure(figsize=(12, 8))
        sorted_data = sorted(ranking_data, key=lambda x: x["individual_scores"].get(metric_name, 0), reverse=True)
        values = [r["individual_scores"].get(metric_name, 0) for r in sorted_data]
        labels = [r["pair"] for r in sorted_data]
        
        plt.barh(labels, values, color='cornflowerblue', edgecolor='black')
        plt.title(f"Ranking by: {metric_name}", fontsize=16, weight='bold')
        plt.xlabel("Score (Higher is Better)", fontsize=12)
        plt.gca().invert_yaxis()
        plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey', axis='x')

        for bar in plt.gca().patches:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f' {bar.get_width():.1f}',
                     va='center', ha='left', fontsize=10)
        
        plt.tight_layout()
        print(f"\n--- Displaying plot for: {metric_name} ---")
        plt.show()

def print_ranking_report_horizontal(ranking_data, metrics):
    print("\n" + "="*150)
    print(" " * 60 + "OVERALL PERFORMANCE RANKING REPORT")
    print("="*150)

    short_headers = {
        "Total Reward": "Reward",
        "Soups Delivered": "Soups",
        "Reward Efficiency": "Effic.",
        "Conflict Count": "Conflict",
        "Time to First Reward": "Time",
        "95th Percentile Euclidean Dist": "Euc. D.",
        "95th Percentile Manhattan Dist": "Manh. D.",
        "95th Percentile Chebyshev Dist": "Cheb. D.",
        "95th Percentile Minkowski p=3 Dist": "Mink. D",
        "DTW Path Distance": "DTW",
    }
    metric_headers = [short_headers.get(m, m) for m in metrics]
    header = f"{'Rank':<6} | {'Agent Pair':<45} | {'Score':<10} | " + " | ".join([f'{h:<8}' for h in metric_headers])
    print(header)
    print("-" * len(header))

    for i, item in enumerate(ranking_data):
        score_str = " | ".join([f"{item['individual_scores'].get(m, 0):>8.1f}" for m in metrics])
        print(f"#{i+1:<5} | {item['pair']:<45} | {item['total_score']:<10.1f} | {score_str}")

    print("="*len(header))
    print("Scores for each metric are normalized from 0 (worst) to 100 (best).")

def plot_and_save_overall_ranking_bar(ranking_data):
    """
    Creates, saves, and displays a bar chart of the overall ranking with updated styling.
    """
    plt.figure(figsize=(14, 9)) # Increased figure size for readability
    
    sorted_data = sorted(ranking_data, key=lambda x: x["total_score"], reverse=True)
    labels = [r["pair"] for r in sorted_data]
    values = [r["total_score"] for r in sorted_data]
    
    plt.barh(labels, values, color='cornflowerblue', edgecolor='black')
    
    # --- UPDATED TITLE AND FONT SIZES ---
    # plt.title("Agent Pair Overall Performance Ranking", fontsize=20, weight='bold')
    plt.xlabel("Total Score (Higher is Better)", fontsize=16)
    plt.ylabel("Agent Pairs", fontsize=16)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    # ------------------------------------

    plt.gca().invert_yaxis()
    plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey', axis='x')
    
    for bar in plt.gca().patches:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f' {bar.get_width():.1f}',
                 va='center', ha='left', fontsize=12)
    
    plt.tight_layout()
    
    output_filename = 'overall_ranking_bar.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n--- Overall ranking bar plot saved as '{output_filename}' ---")
    
    print("--- Displaying overall ranking plot ---")
    plt.show()

def plot_and_save_rank_distribution_scatter(ranking_data, metrics):
    """
    Creates, saves, and displays the rank distribution scatter plot with updated styling.
    """
    if not ranking_data:
        return
        
    pairs = sorted([item['pair'] for item in ranking_data])
    colors = plt.get_cmap('tab20', len(pairs))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', '<', '>', 'X']
    
    style_map = {pair: {'color': colors(i), 'marker': markers[i % len(markers)]} for i, pair in enumerate(pairs)}

    plot_data = []
    for metric in metrics:
        sorted_by_metric = sorted(ranking_data, key=lambda x: x['individual_scores'].get(metric, 0), reverse=True)
        for i, item in enumerate(sorted_by_metric):
            rank = i + 1
            plot_data.append({'metric': metric, 'rank': rank, 'pair': item['pair']})

    plt.figure(figsize=(16, 12)) # Increased figure size
    
    for item in plot_data:
        style = style_map[item['pair']]
        y_val = metrics.index(item['metric']) + np.random.uniform(-0.1, 0.1)
        plt.scatter(item['rank'], y_val, color=style['color'], marker=style['marker'], s=250, alpha=0.8, edgecolors='black')

    legend_elements = [Line2D([0], [0], marker=style_map[pair]['marker'], color=style_map[pair]['color'], label=pair,
                              linestyle='None', markersize=12) for pair in pairs]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title="Agent Pairs", fontsize=14)
    
    # --- UPDATED FONT SIZES ---
    plt.yticks(range(len(metrics)), metrics, fontsize=12)
    plt.xlabel("Performance Rank (1 is best)", fontsize=16)
    # plt.title("System Performance Rank Across Metrics", fontsize=20, weight='bold')
    plt.tick_params(axis='x', labelsize=12)
    # --------------------------

    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.xticks(np.arange(1, len(pairs) + 1))
    plt.gca().invert_yaxis()
    
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    
    output_filename = 'rank_distribution_scatter.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n--- Rank distribution scatter plot saved as '{output_filename}' ---")

    print("--- Displaying rank distribution scatter plot ---")
    plt.show()


if __name__ == "__main__":
    FOLDER_TO_ANALYZE = "./game_trajectories/Cramped Room"
    EPISODES_TO_USE = range(1, 21)

    print(f"Analyzing episodes {min(EPISODES_TO_USE)}-{max(EPISODES_TO_USE)} from folder: '{FOLDER_TO_ANALYZE}'")

    try:
        aggregated_stats = compute_aggregated_pair_episode_data(
            FOLDER_TO_ANALYZE,
            episodes_to_average=EPISODES_TO_USE
        )

        if not aggregated_stats:
            print("\nNo data was found. Please check the folder path and file names.")
        else:
            final_ranks, used_metrics = calculate_and_rank_metrics(aggregated_stats)

            print_ranking_report_horizontal(final_ranks, used_metrics)
            
            plot_all_rankings_individually(final_ranks, used_metrics)
            
            plot_and_save_overall_ranking_bar(final_ranks)

            plot_and_save_rank_distribution_scatter(final_ranks, used_metrics)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\nPlease ensure the `FOLDER_TO_ANALYZE` variable is set correctly.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")