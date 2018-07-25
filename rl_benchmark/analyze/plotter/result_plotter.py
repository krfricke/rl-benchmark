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

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from rl_benchmark.analyze.transform import rewards_by_episode, rewards_by_timestep, rewards_by_second, \
    to_timeseries


class ResultPlotter(object):
    def __init__(self):
        self.benchmarks = list()
        self.palette = None

    def make_palette(self):
        if not self.palette:
            self.palette = sns.color_palette("husl", len(self.benchmarks))

    def add_benchmark(self, benchmark_data, name):
        self.benchmarks.append((benchmark_data, name))

    def plot_reward_by_episode(self, ax=None):
        self.make_palette()

        full_data = pd.DataFrame()
        for idx, (benchmark_data, name) in enumerate(self.benchmarks):
            plot_data = to_timeseries(benchmark_data, x_label="Episode", y_label="Average Episode Reward",
                                    target=rewards_by_episode, cut_x=benchmark_data.min_x('episodes'), smooth=10)
            plot_data['Benchmark'] = name
            full_data = full_data.append(plot_data)

        plot = sns.tsplot(data=full_data, time="Episode", value="Average Episode Reward", unit="experiment",
                          condition='Benchmark', ax=ax, ci=[68, 95], color=self.palette)

        return plot

    def plot_reward_by_timestep(self, ax=None):
        self.make_palette()

        full_data = pd.DataFrame()
        for idx, (benchmark_data, name) in enumerate(self.benchmarks):
            plot_data = to_timeseries(benchmark_data, x_label="Time step", y_label="Average Episode Reward",
                                    target=rewards_by_timestep, cut_x=benchmark_data.min_x('timesteps'), smooth=10)
            plot_data['Benchmark'] = name
            full_data = full_data.append(plot_data)

        plot = sns.tsplot(data=full_data, time="Time step", value="Average Episode Reward", unit="experiment",
                          condition='Benchmark', ax=ax, ci=[68, 95], color=self.palette)

        return plot

    def plot_reward_by_second(self, ax=None):
        self.make_palette()

        full_data = pd.DataFrame()
        for idx, (benchmark_data, name) in enumerate(self.benchmarks):
            plot_data = to_timeseries(benchmark_data, x_label="Second", y_label="Average Episode Reward",
                                    target=rewards_by_second, cut_x=benchmark_data.min_x('seconds'), smooth=10)
            plot_data['Benchmark'] = name
            full_data = full_data.append(plot_data)

        plot = sns.tsplot(data=full_data, time="Second", value="Average Episode Reward", unit="experiment",
                          condition='Benchmark', ax=ax, ci=[68, 95], color=self.palette)

        return plot