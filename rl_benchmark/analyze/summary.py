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

"""
Summary functions for benchmarks.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


if __name__ == '__main__':
    def solved_after(benchmark_data, reward_threshold, minimum_episodes=1, report_steps=False):
        """
        Return average amount of episodes after which an episode is considered solved.

        Args:
            benchmark_data: benchmark_data or experiment_data object
            reward_threshold: reward threshold considered as solved
            minimum_episodes: minimum consecutive episodes a reward threshold must be reached
            report_steps: boolean whether to return timesteps instead of episodes.

        Returns: tuple: (average episode/step count, sd)

        """
        # Also accept a single experiment_data object
        if not isinstance(benchmark_data, list):
            benchmark_data = [benchmark_data]

        array = np.array([experiment_data['results']['episode_rewards'] for experiment_data in benchmark_data])
        # TODO: implement this..
        raise NotImplementedError


def average_reward(benchmark_data, episodes=0):
    """
    Return average award over the last `episodes` episodes.
    Args:
        benchmark_data: benchmark_data or experiment_data object
        episodes: count of episodes to include. 0 means include all.

    Returns: tuple (average reward, sd)

    """
    if not isinstance(benchmark_data, list):
        benchmark_data = [benchmark_data]

    array = np.array([experiment_data['results']['episode_rewards'][-episodes:] for experiment_data in benchmark_data])

    return array.mean(), array.std()
