import matplotlib.pyplot as plt

import json
import numpy as np
from itertools import permutations
from scipy.spatial.distance import euclidean
from dtw import dtw


class TeamComparator:
    def __init__(self, delta=0.05):
        self.delta = delta
        self.epsilon = None
        self.ref_trajectories = []
        self.ref_distances = []  # Store reference pairwise distances
        self.test_distances = []  # Store test vs reference distances
        self.calibration_scores = []



    def _load_team_data(self, filepath):
        """Load and validate team trajectory data from JSON file"""
        with open(filepath) as f:
            data = json.load(f)

        trajectories = data['ep_observations']

        # Ensure each state in trajectories is a dictionary
        processed = []
        for traj in trajectories:
            valid_traj = []
            for state in traj:
                # Unpack state if it's inside a singleton list
                if isinstance(state, list) and len(state) == 1:
                    state = state[0]
                # Validate state structure
                if not isinstance(state, dict) or 'players' not in state:
                    raise ValueError("Invalid state format in trajectory")
                valid_traj.append(state)
            processed.append(valid_traj)

        return processed

    def _state_distance(self, s1, s2):
        """Custom distance metric between two game states"""
        # if len(s1[0]['players']) != len(s2[0]['players']):
        #     return float('inf')

        min_dist = float('inf')
        num_players = len(s1[0]['players'])
        for perm in permutations(range(num_players)):
            current_dist = 0
            for i, j in zip(range(num_players), perm):
                p1 = s1[0]['players'][i]
                p2 = s2[0]['players'][j]

                pos_dist = euclidean(p1['position'], p2['position'])
                ori_dist = euclidean(p1['orientation'], p2['orientation'])

                h1 = p1.get('held_object')
                h2 = p2.get('held_object')
                held_dist = 0
                if (h1 is None) != (h2 is None):
                    held_dist = 1
                elif h1 and h2 and (h1['name'] != h2['name']):
                    held_dist = 1

                current_dist += pos_dist + ori_dist + held_dist

            if current_dist < min_dist:
                min_dist = current_dist

        return min_dist

    def _trajectory_distance(self, t1, t2):
        """Dynamic Time Warping distance between trajectories"""
        alignment = dtw(t1, t2, dist_method=self._state_distance)
        # alignment = dtw(t1, t2, dist=self._state_distance)

        return alignment.distance

    def fit_reference(self, ref_file):
        """Fit reference trajectories using conformal prediction"""
        self.ref_trajectories = self._load_team_data(ref_file)

        # Calculate leave-one-out conformity scores
        self.calibration_scores = []
        for i, traj in enumerate(self.ref_trajectories):
            other_trajs = [t for j, t in enumerate(self.ref_trajectories) if j != i]
            min_dist = min(self._trajectory_distance(traj, t) for t in other_trajs)
            self.calibration_scores.append(min_dist)

        return self

    def compare_team(self, test_file):
        """Compare test team using conformal prediction p-values"""
        test_trajectories = self._load_team_data(test_file)
        alerts = []
        p_values = []

        for test_traj in test_trajectories:
            test_score = min(self._trajectory_distance(test_traj, ref_traj)
                             for ref_traj in self.ref_trajectories)
            pval = (1 + sum(score <= test_score for score in self.calibration_scores)) \
                   / (1 + len(self.calibration_scores))
            p_values.append(pval)
            alerts.append(1 if pval < self.delta else 0)

        return alerts, p_values

    def plot_distances(self, save_path=None):
        """Visualize distance distributions and threshold"""
        plt.figure(figsize=(10, 6))

        # Plot reference distances
        plt.hist(self.ref_distances, bins=30, alpha=0.7,
                 label='Reference Team Pairwise Distances')

        # Plot test distances if available
        if self.test_distances:
            plt.hist(self.test_distances, bins=30, alpha=0.7,
                     label='Test Team vs Reference Distances')

        # Add threshold line
        if self.epsilon is not None:
            plt.axvline(self.epsilon, color='red', linestyle='--',
                        label=f'Threshold (ε = {self.epsilon:.2f})')

        plt.xlabel('DTW Distance')
        plt.ylabel('Frequency')
        plt.title('Distance Distribution and Conformance Threshold')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


# Modified Usage Example
if __name__ == "__main__":
    comparator = TeamComparator(delta=0.05)
    epsilon = comparator.fit_reference(
        "Cramped_room/Human-aware PPO agent vs Human Keyboard Input 60 sec Cramped Room.json")

    alerts = comparator.compare_team(
        "Cramped_room/Human-aware PPO agent vs Human-aware PPO agent 60 sec Cramped Room.json")

    # Generate and show the histogram
    comparator.plot_distances()
