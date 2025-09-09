# Overcooked-AI 🧑‍🍳🤖

> **Note**
> This is a fork of the original [HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) repository, adapted for the research detailed below.

## About This Project: Risk Analysis of Player Substitutions

The code in this repository was used to investigate the impact of player substitutions on the coordination and dynamics of multi-agent teams. By leveraging risk analysis, we can formally detect when switching a player significantly disrupts established collaboration patterns.

**Abstract:**

The substitution of players in an existing multi-agent team often disrupts established coordination patterns, leading to outcomes that range from rapid adaptation to systemic breakdowns. In this work, we leverage risk analysis to formally detect when switching a player significantly impacts team dynamics. By repeatedly simulating gameplay and systematically comparing team behavior before and after player substitutions, we identify which substituted players induce the most substantial changes in collaboration patterns. Our analysis reveals that agents trained with human-aware objectives tend to maintain smoother coordination after switches, while those based solely on human demonstrations or self-play lead to greater spatial and temporal disruption during partner transitions. We demonstrate the effectiveness of our approach using the Overcooked benchmark.


<p align="center">
  <img src="./images/layouts.gif" width="100%"> 
  <i>5 of the available layouts. New layouts are easy to hardcode or generate programmatically.</i>
</p>

## Introduction 🥘

Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/).

The goal of the game is to deliver soups as fast as possible. Each soup requires placing up to 3 ingredients in a pot, waiting for the soup to cook, and then having an agent pick up the soup and delivering it. The agents should split up tasks on the fly and coordinate effectively in order to achieve high reward.

You can **try out the game [here](https://humancompatibleai.github.io/overcooked-demo/)** (playing with some previously trained DRL agents). To play with your own trained agents using this interface, or to collect more human-AI or human-human data, you can use the code [here](https://github.com/HumanCompatibleAI/overcooked_ai/tree/master/src/overcooked_demo). You can find some human-human and human-AI gameplay data already collected [here](https://github.com/HumanCompatibleAI/overcooked_ai/tree/master/src/human_aware_rl/static/human_data).

### **Analysis (`/analysis`)** 📊

This directory contains scripts for processing the raw simulation data from the `game_trajectories` folder to quantitatively evaluate agent performance and team dynamics.

#### **`overall_ranking.py`**

This is the primary script for performing the risk analysis and performance evaluation. It systematically processes the JSON trajectory files to rank each agent pair based on a wide range of metrics.

**Key Functionality:**

* **Data Parsing:** Reads game data from multiple episodes for each unique agent pair (e.g., "Human-aware PPO vs. Self-Play").
* **Metric Calculation:** Computes a comprehensive set of metrics to evaluate both task success and coordination quality. These include:
    * **Performance Metrics:** Total Reward, Soups Delivered, and Reward Efficiency.
    * **Coordination Metrics:** Coordination Conflicts (agents attempting to interact with the same space) and **Dynamic Time Warping (DTW)** to measure the similarity of agent movement paths.
    * **Spatial Disruption Metrics:** Calculates the 95th percentile of various distances (Euclidean, Manhattan, etc.) between agents.
* **Ranking System:** Normalizes each metric to a score from 0 (worst) to 100 (best) and calculates a total composite score to produce a final ranking for each agent pair.

**How to Run:**
Simply execute the script from your terminal:
```bash
python analysis/overall_ranking.py
```

**Outputs:**
Running the script generates several outputs:
1.  A detailed **ranking report** printed directly to the terminal.
2.  A series of saved plots (`.png` files) for visualization, including:
    * `overall_ranking_bar.png`: A bar chart showing the final composite score and rank for each pair.
    * `rank_distribution_scatter.png`: A scatter plot that visualizes how each team ranks across all the different individual metrics.
    * `all_metrics_rankings_combined.png`: A combined figure showing the rankings for every single metric.

## Installation ☑️

### Installing from PyPI 🗜

You can install the pre-compiled wheel file using pip.
`pip install overcooked-ai`
Note that PyPI releases are stable but infrequent. For the most up-to-date development features, build from source. We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) to install the package, so that you can use the provided lockfile to ensure no minimal package version issues.


### Building from source 🔧

Clone the repo
`git clone https://github.com/HumanCompatibleAI/overcooked_ai.git`

Using uv (recommended):
`uv venv`
`uv sync`


### Verifying Installation 📈

When building from source, you can verify the installation by running the Overcooked unit test suite. The following commands should all be run from the `overcooked_ai` project root directory:

`python testing/overcooked_test.py`

If you're thinking of using the planning code extensively, you should run the full testing suite that verifies all of the Overcooked accessory tools (this can take 5-10 mins):
`python -m unittest discover -s testing/ -p "*_test.py"`

See this [notebook](Overcooked%20Tutorial.ipynb) for a quick guide on getting started using the environment.

## Code Structure Overview 🗺

`overcooked_ai_py` contains:

`mdp/`:
- `overcooked_mdp.py`: main Overcooked game logic
- `overcooked_env.py`: environment classes built on top of the Overcooked mdp
- `layout_generator.py`: functions to generate random layouts programmatically

`agents/`:
- `agent.py`: location of agent classes
- `benchmarking.py`: sample trajectories of agents (both trained and planners) and load various models

`planning/`:
- `planners.py`: near-optimal agent planning logic
- `search.py`: A* search and shortest path logic

`overcooked_demo` contains:

`server/`:
- `app.py`: The Flask app
- `game.py`: The main logic of the game. State transitions are handled by overcooked.Gridworld object embedded in the game environment
- `move_agents.py`: A script that simplifies copying checkpoints to [agents](src/overcooked_demo/server/static/assets/agents/) directory. Instruction of how to use can be found inside the file or by running `python move_agents.py -h`

`up.sh`: Shell script to spin up the Docker server that hosts the game

`human_aware_rl` contains (NOTE: this is not supported anymore, see bottom of the README for more info):

`ppo/`:
- `ppo_rllib.py`: Primary module where code for training a PPO agent resides. This includes an rllib compatible wrapper on `OvercookedEnv`, utilities for converting rllib `Policy` classes to Overcooked `Agent`s, as well as utility functions and callbacks
- `ppo_rllib_client.py` Driver code for configuing and launching the training of an agent. More details about usage below
- `ppo_rllib_from_params_client.py`: train one agent with PPO in Overcooked with variable-MDPs
- `ppo_rllib_test.py` Reproducibility tests for local sanity checks
- `run_experiments.sh` Script for training agents on 5 classical layouts
- `trained_example/` Pretrained model for testing purposes

`rllib/`:
- `rllib.py`: rllib agent and training utils that utilize Overcooked APIs
- `utils.py`: utils for the above
- `tests.py`: preliminary tests for the above

`imitation/`:
- `behavior_cloning_tf2.py`: Module for training, saving, and loading a BC model
- `behavior_cloning_tf2_test.py`: Contains basic reproducibility tests as well as unit tests for the various components of the bc module.

`human/`:
- `process_data.py` script to process human data in specific formats to be used by DRL algorithms
- `data_processing_utils.py` utils for the above

`utils.py`: utils for the repo


## Raw Data :

The raw data used during BC training is >100 MB, which makes it inconvenient to distribute via git. The code uses pickled dataframes for training and testing, but in case one needs to original data it can be found [here](https://drive.google.com/drive/folders/1aGV8eqWeOG5BMFdUcVoP2NHU_GFPqi57?usp=share_link)