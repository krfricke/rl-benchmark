# Copyright 2018 The RLgraph project. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from rl_benchmark.util import n_step_average


def rewards_by_episode(rewards, cut_x=1e12, *args, **kwargs):
    episodes = np.arange(len(rewards))
    episodes, rewards = episodes[episodes < cut_x], rewards[episodes < cut_x]

    if cut_x > 200:
        episodes = np.linspace(0, cut_x, 200)
        rewards = n_step_average(rewards, 200)

    return episodes, rewards


def rewards_by_timestep(rewards, timesteps, cut_x=1e12, *args, **kwargs):
    timesteps, rewards = timesteps[timesteps < cut_x], rewards[timesteps < cut_x]

    if cut_x > 200:
        timesteps = np.linspace(0, cut_x, 200)
        rewards = n_step_average(rewards, 200)

    return timesteps, rewards


def rewards_by_second(rewards, seconds=None, cut_x=1e12, *args, **kwargs):
    cut_x = int(cut_x)

    seconds, rewards = seconds[seconds < cut_x], rewards[seconds < cut_x]

    if cut_x > len(rewards):
        seconds = np.linspace(0, cut_x, 200)
        rewards = n_step_average(rewards, 200)
    else:
        seconds = np.linspace(0, cut_x, cut_x)
        rewards = n_step_average(rewards, cut_x)

    return seconds, rewards


def to_timeseries(benchmark_data, x_label='Episode', y_label='Average Episode Reward',
                  target=rewards_by_episode, cut_x=1e12, smooth=0):
    """
    Convert benchmark data to timeseries data, plottable my mathplotlib.

    Args:
        benchmark_data: BenchmarkData object
        x_label: label for the x axis (time)
        y_label: label for the y axis (values)
        target: callback returning processed x and y values
        cut_x: maximum x value to cut (passed to target)
        smooth: used to np.ewm(span=smooth) (smooth curve)

    Returns: pd.DataFrame

    """
    data_experiments, data_times, data_values = [], [], []

    for experiment_id, experiment_data in enumerate(benchmark_data):
        extended_results = experiment_data.extended_results()

        if smooth > 0:
            extended_results['rewards'] = np.array(pd.Series(extended_results['rewards']).ewm(span=smooth).mean())

        x, y = target(cut_x=cut_x, **extended_results)

        data_times.extend(x)
        data_values.extend(y)
        data_experiments.extend([experiment_id] * len(x))

    return pd.DataFrame({'experiment': data_experiments, x_label: data_times, y_label: data_values})
