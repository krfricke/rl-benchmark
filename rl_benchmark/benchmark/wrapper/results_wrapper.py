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
Results wrapper class.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

from rl_benchmark.benchmark.wrapper.environment_wrapper import EnvironmentWrapper


class ResultsWrapper(EnvironmentWrapper):
    def __init__(self, env):
        super(ResultsWrapper, self).__init__(env)

        self.episode = 1
        self.timestep = 0

        self.episode_rewards = list()
        self.episode_timesteps = list()
        self.episode_times = list()

        self.episode_start_time = time.time()
        self.episode_timestep = 0
        self.episode_reward = 0

    def reset(self):
        # Only reset episode statistics:
        self.episode_start_time = time.time()
        self.episode_timestep = 0
        self.episode_reward = 0

        return super(ResultsWrapper, self).reset()

    def get_results(self):
        results = dict(
            initial_reset_time=0,
            episode_rewards=self.episode_rewards,
            episode_timesteps=self.episode_timesteps,
            episode_end_times=self.episode_times
        )

        return results
