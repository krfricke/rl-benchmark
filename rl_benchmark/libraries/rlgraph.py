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
RLgraph benchmarking.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
import time

from tensorflow import __version__ as tensorflow_version

from rlgraph import __version__ as rlgraph_version
from copy import copy

from rl_benchmark.benchmark.runner.benchmark_runner import BenchmarkRunner
from rl_benchmark.benchmark.wrapper.results_wrapper import ResultsWrapper

from rlgraph.agents import Agent
from rlgraph.environments import OpenAIGymEnv
from rlgraph.execution.single_threaded_worker import SingleThreadedWorker


class RLgraphEnvironmentWrapper(ResultsWrapper):
    """
    RLgraph's environment don't support end-of-episode callbacks by default, so we introduce them by wrapping
    the environment objects.
    """
    def step(self, *args, **kwargs):
        state, reward, terminal, info = self.env.step(*args, **kwargs)

        self.episode_timestep += 1
        self.episode_reward += reward

        if terminal:
            time_passed = time.monotonic() - self.episode_start_time

            self.episode_timesteps.append(self.episode_timestep)
            self.episode_rewards.append(self.episode_reward)
            self.episode_times.append(time_passed)

            self.call_episode_end_callbacks()

        return state, reward, terminal, info


class RLgraphBenchmarkRunner(BenchmarkRunner):
    rl_library = 'rlgraph'
    rl_library_version = rlgraph_version
    rl_backend = 'tensorflow'
    rl_backend_version = tensorflow_version

    def __init__(self, config=None, config_folder=None, output_folder='/tmp'):
        super(RLgraphBenchmarkRunner, self).__init__(config, config_folder, output_folder)

        self.environment_callback = None

        # This file is WIP

    def make_environment(self):
        """
        Create environment.

        Returns: environment

        """
        (environment_class, environment_args, environment_kwargs) = self.environment_callback

        if isinstance(environment_class, str):
            if self.environment_domain == 'openai_gym':
                environment_class = OpenAIGymEnv
            else:
                raise NotImplementedError("No method for creating non-gym environment.")

        environment = environment_class(*environment_args, **environment_kwargs)

        return environment

    def run_experiment(self, environment, experiment_num=0):
        environment = RLgraphEnvironmentWrapper(environment)
        environment.add_episode_end_callback(self.episode_finished, environment, runner_id=1)

        config = copy(self.config)

        max_episodes = config.pop('max_episodes', None)
        max_timesteps = config.pop('max_timesteps', None)
        max_episode_timesteps = config.pop('max_episode_timesteps')

        agent = Agent.from_spec(
            spec=config,
            state_space=environment.state_space,
            action_space=environment.action_space,
        )

        if experiment_num == 0 and self.load_model_file:
            logging.info("Loading model data from file: {}".format(self.load_model))
            agent.load_model(self.load_model_file)

        runner = SingleThreadedWorker(
            agent=agent,
            environment=environment
        )

        environment.reset()
        agent.reset_buffers()

        if max_timesteps:
            runner.execute_timesteps(num_timesteps=max_timesteps, max_timesteps_per_episode=max_episode_timesteps)
        else:
            runner.execute_episodes(num_episodes=max_episodes, max_timesteps_per_episode=max_episode_timesteps)

        return dict(
            initial_reset_time=0,
            episode_rewards=runner.episode_rewards,
            episode_timesteps=runner.episode_steps,
            episode_end_times=runner.episode_durations
        )
