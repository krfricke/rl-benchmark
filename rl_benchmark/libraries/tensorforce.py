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
Tensorforce benchmarking.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging

from copy import copy

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

from tensorflow import __version__ as tensorflow_version
from tensorforce import __version__ as tensorforce_version

from rl_benchmark.benchmark.runner.benchmark_runner import BenchmarkRunner


class TensorForceBenchmarkRunner(BenchmarkRunner):
    rl_library = 'tensorforce'
    rl_library_version = tensorforce_version
    rl_backend = 'tensorflow'
    rl_backend_version = tensorflow_version

    def __init__(self, config=None, config_folder=None, output_folder='/tmp'):
        super(TensorForceBenchmarkRunner, self).__init__(config, config_folder, output_folder)

        self.environment_callback = None

    def make_environment(self):
        """
        Create environment.

        Returns: environment

        """
        (environment_class, environment_args, environment_kwargs) = self.environment_callback

        if isinstance(environment_class, str):
            if self.environment_domain == 'openai_gym':
                environment_class = OpenAIGym
            else:
                raise NotImplementedError("No method for creating non-gym environment.")

        environment = environment_class(*environment_args, **environment_kwargs)

        return environment

    def run_experiment(self, environment, experiment_num=0):
        config = copy(self.config)

        max_episodes = config.pop('max_episodes', None)
        max_timesteps = config.pop('max_timesteps', None)
        max_episode_timesteps = config.pop('max_episode_timesteps')

        network_spec = config.pop('network')

        agent = Agent.from_spec(
            spec=config,
            kwargs=dict(
                states=environment.states,
                actions=environment.actions,
                network=network_spec
            )
        )

        if experiment_num == 0 and self.history_data:
            logging.info("Attaching history data to runner")
            history_data = self.history_data
        else:
            history_data = None

        if experiment_num == 0 and self.load_model_file:
            logging.info("Loading model data from file: {}".format(self.load_model))
            agent.load_model(self.load_model_file)

        runner = Runner(
            agent=agent,
            environment=environment,
            repeat_actions=1,
            history=history_data
            # save_path=args.model,
            # save_episodes=args.save_model
        )

        environment.reset()
        agent.reset()

        runner.run(episodes=max_episodes, timesteps=max_timesteps, max_episode_timesteps=max_episode_timesteps,
                   episode_finished=self.episode_finished)

        return dict(
            initial_reset_time=0,
            episode_rewards=runner.episode_rewards,
            episode_timesteps=runner.episode_timesteps,
            episode_end_times=runner.episode_times
        )
