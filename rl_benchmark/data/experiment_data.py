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

from rl_benchmark.util import hash_object


class ExperimentData(dict):

    def hash(self):
        """
        Calculate experiment_hash, benchmark_hash, and config_hash to enable quick lookup of equal or similar
        benchmarks.

        Returns: tuple of experiment_hash, benchmark_hash, and config_hash

        """
        # Hash configuration to quickly find benchmarks with equal configurations
        config = self['config']
        config_hash = hash_object(config)

        # Benchmark hash identifies runs on the same config, environment and RL library/backend version
        metadata = self['metadata']
        results = self['results']

        benchmark_hash = hash_object([
            config_hash,
            metadata['environment_domain'],
            metadata['environment_name'],
            metadata['rl_library'],
            metadata['rl_library_version'],
            metadata['rl_backend'],
            metadata['rl_backend_version']
        ])

        # Experiment hash identifies a specific benchmark run
        experiment_hash = hash_object([
            config_hash,
            benchmark_hash,
            results['episode_rewards'],
            results['episode_timesteps'],
            results['episode_end_times']
        ])

        return experiment_hash, benchmark_hash, config_hash


    def extended_results(self):
        rewards = np.array(self['results']['episode_rewards'])  # turn rewards into numpy array

        episode_timesteps = np.array(self['results']['episode_timesteps'])  # episode lengths
        timesteps = np.cumsum(episode_timesteps)  # cumulative episode lengths as timesteps

        seconds = np.cumsum(self['results']['episode_end_times'])  # cumulative times in seconds

        return dict(
            rewards=rewards,
            episode_timesteps=episode_timesteps,
            timesteps=timesteps,
            seconds=seconds,
            episodes=np.arange(len(rewards))
        )
