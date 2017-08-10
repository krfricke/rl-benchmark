# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
TensorForce benchmark result analysis and formatting.

Usage:

```bash
python results.py [--output output] [--input <file> <name>] [--input <file> <name> ...]
```

`input` expects two parameters. `file` points to a pickle file (pkl) containing experiment data (e.g. created by
running `benchmark.py`). `name` is a string containing the label for the plot. You can state multiple input files.

`output` is an optional parameter to set the output (png) file. If omitted, output will be saved as `./output.png`.

The resulting output file is a PNG image containing plots for rewards by episodes and rewards by timesteps.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig()


def n_step_average(data, n):
    """
    Average data over n steps.

    Args:
        data: np.array containing the ata
        n: steps to average over

    Returns: np.array of size n containing the average of the respective bins

    """
    if len(data) < n:
        n = len(data)

    cut = data[0:len(data)-len(data)%n]  # cut array so it's divisible by n
    return np.mean(cut.reshape(-1, len(data) // n), axis=1)


def rewards_by_episodes(rewards, timesteps=None, lengths=None, seconds=None, cut_x=1e12):
    episodes = np.arange(len(rewards))
    episodes, rewards = episodes[episodes < cut_x], rewards[episodes < cut_x]

    if cut_x > 200:
        episodes = np.linspace(0, cut_x, 200)
        rewards = n_step_average(rewards, 200)

    return episodes, rewards


def rewards_by_timesteps(rewards, timesteps, lengths=None, seconds=None, cut_x=1e12):
    timesteps, rewards = timesteps[timesteps < cut_x], rewards[timesteps < cut_x]

    if cut_x > 200:
        timesteps = np.linspace(0, cut_x, 200)
        rewards = n_step_average(rewards, 200)

    return timesteps, rewards


def rewards_by_seconds(rewards, timesteps, lengths=None, seconds=None, cut_x=1e12):
    cut_x = int(cut_x)

    seconds, rewards = seconds[seconds < cut_x], rewards[seconds < cut_x]

    if cut_x > len(rewards):
        seconds = np.linspace(0, cut_x, 200)
        rewards = n_step_average(rewards, 200)
    else:
        seconds = np.linspace(0, cut_x, cut_x)
        rewards = n_step_average(rewards, cut_x)

    return seconds, rewards


def to_timeseries(full_data, x_label='Episode', y_label='Average Episode Reward',
                  target=rewards_by_episodes, cut_x=1e12, smooth=0):
    """
    Convert benchmark data to timeseries data, plottable my mathplotlib.

    Args:
        full_data: list of tuples (of lists) for each experiment (lengths, timesteps (cumulated), rewards)
        x_label: label for the x axis (time)
        y_label: label for the y axis (values)
        target: callback returning processed x and y values
        cut_x: maximum x value to cut (passed to target)
        smooth: used to np.ewm(span=smooth) (smooth curve)

    Returns: pd.DataFrame

    """
    data_experiments, data_times, data_values = [], [], []

    for experiment_id, (lengths, timesteps, seconds, rewards) in enumerate(full_data):
        if smooth > 0:
            rewards = np.array(pd.Series(rewards).ewm(span=smooth).mean())

        x, y = target(rewards, timesteps=timesteps, seconds=seconds, lengths=lengths, cut_x=cut_x)

        data_times.extend(x)
        data_values.extend(y)
        data_experiments.extend([experiment_id] * len(x))

    return pd.DataFrame({'experiment': data_experiments, x_label: data_times, y_label: data_values})


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', action='append', nargs=2, metavar=('file', 'name'),
                        help="Input file(s): <file> <name>")
    parser.add_argument('-o', '--output', default="output.png", help="output file (image png)")

    args = parser.parse_args()

    if len(args.input) < 1:
        raise ValueError("Please state at least one input file and name.")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # load input files into data dict
    data = dict()
    for (filename, name) in args.input:
        logger.info("Loading {} ({})".format(filename, name))
        with open(os.path.join(os.getcwd(), filename), "rb") as f:
            data[name] = pickle.load(f)

    # plot_cols = min(len(data.keys()) * 2, 4)
    # plot_rows = (len(data.keys()) * 2 + plot_cols - 1) // plot_cols
    plot_cols = 3
    plot_rows = 1

    figure, axes = plt.subplots(ncols=plot_cols, nrows=plot_rows, figsize=(plot_cols * 6, plot_rows * 6))

    # figure = plt.figure()

    colors = ['cyan', 'magenta', 'yellow', 'blue', 'green', 'red', 'black']

    ax_legends = [list() for ax in axes]
    # iterate through all input datasets
    for data_id, (name, benchmark_data) in enumerate(data.iteritems()):
        color = colors[data_id % len(colors)]
        patch = mpatches.Patch(color=color, label=name)

        logger.info("Plotting average rewards for {} in {}".format(name, color))

        least_episodes = least_timesteps = least_seconds = int(1e12)
        full_data = list()
        logger.info("Found {} experiments.".format(len(benchmark_data)))
        for experiment_data in benchmark_data:
            lengths = np.array(experiment_data['episode_lengths'])  # episode lengths
            timesteps = np.cumsum(lengths)  # cumulative episode lengths as timesteps
            rewards = np.array(experiment_data['episode_rewards'])  # turn rewards into numpy array
            seconds = np.cumsum(experiment_data['episode_end_times'])  # cumulative times in seconds

            full_data.append((lengths, timesteps, seconds, rewards))

            least_episodes = min(least_episodes, len(rewards))
            least_timesteps = min(least_timesteps, timesteps[-1])
            least_seconds = min(least_seconds, seconds[-1])

        # TODO: Maybe limit all input datasets to least_episodes / least_timesteps?

        ax_index = -1

        # Plot average rewards by episodes
        re_plot = to_timeseries(full_data, x_label="Episode", y_label="Average Episode Reward",
                                target=rewards_by_episodes, cut_x=least_episodes, smooth=10)

        ax_index += 1
        ax = axes[ax_index]

        plot = sns.tsplot(data=re_plot, time="Episode", value="Average Episode Reward", unit="experiment",
                          ax=ax, ci=[68, 95], color=color)

        ax_legends[ax_index].append(patch)

        figure.add_subplot(plot)

        # Plot average rewards by timesteps
        rt_plot = to_timeseries(full_data, x_label="Timestep", y_label="Average Episode Reward",
                                target=rewards_by_timesteps, cut_x=least_timesteps, smooth=10)

        ax_index += 1
        ax = axes[ax_index]

        plot = sns.tsplot(data=rt_plot, time="Timestep", value="Average Episode Reward", unit="experiment",
                          ax=ax, ci=[68, 95], color=color)

        ax_legends[ax_index].append(patch)

        figure.add_subplot(plot)

        # Plot average rewards by seconds
        rs_plot = to_timeseries(full_data, x_label="Seconds", y_label="Average Episode Reward",
                                target=rewards_by_seconds, cut_x=least_seconds, smooth=10)

        ax_index += 1
        ax = axes[ax_index]

        plot = sns.tsplot(data=rs_plot, time="Seconds", value="Average Episode Reward", unit="experiment",
                          ax=ax, ci=[68, 95], color=color)

        ax_legends[ax_index].append(patch)

        figure.add_subplot(plot)


    for ax_index, ax in enumerate(axes):
        ax.legend(handles=ax_legends[ax_index])

    logger.info("Saving figure to {}".format(args.output))
    figure.savefig(args.output)


if __name__ == '__main__':
    main()
