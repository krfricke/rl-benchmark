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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import numpy as np
import os
import pickle

from copy import copy
from six.moves import xrange
from tensorflow import __version__ as tensorflow_version

from tensorforce.agents import agents
from tensorforce import Configuration, __version__ as tensorforce_version
from tensorforce.execution import Runner


class BenchmarkRunner(object):
    def __init__(self, config=None, config_folder=None, output_folder='/tmp'):
        self.config = config
        self.config_folder = config_folder
        self.output_folder = output_folder

        self.output_filename = None
        self.history_data = None
        self.load_model_file = None

        self.report_episodes = 10

        self.save_history_file = None
        self.save_history_episodes = 0

        self.save_model_file = None
        self.save_model_episodes = 0

        self.current_run_results = None

        self.environment_callback = None
        self.environment_domain = 'user'
        self.environment_name = None

    def load_config(self, filename):
        """
        Load config from file. Either state a file from the config_folder (with or without file suffix),
        or supply full file path.

        Args:
            filename: Filename or full file path

        Returns: Boolean

        """

        possible_config_file_paths = [
            os.path.join(os.getcwd(), filename)  # first check absolute path
        ]

        if self.config_folder:
            possible_config_file_paths += [
                os.path.join(self.config_folder, '{}'.format(filename)),  # check with user-supplied file suffix
                os.path.join(self.config_folder, '{}.json'.format(filename))  # check with json suffix
            ]

        for possible_config_file_path in possible_config_file_paths:
            if not os.path.exists(possible_config_file_path):
                logging.debug("Possible config file does not exist: {}".format(possible_config_file_path))
                continue

            logging.debug("Found config file at {}".format(possible_config_file_path))

            if self.config:
                logging.warning("Overriding existing configuration")

            self.config = Configuration.from_json(possible_config_file_path)
            return True

        return False

    def set_environment(self, environment_class, *args, **kwargs):
        """
        Set environment class and store as callback

        Args:
            environment_class:
            *args: arguments to pass to environment class constructor
            **kwargs: keyword arguments to pass to environment class constructor

        Returns:

        """
        self.environment_callback = (environment_class, args, kwargs)

        if environment_class.__name__ == 'OpenAIGym':
            self.environment_domain = 'openai_gym'
            self.environment_name = args[0]
        else:
            self.environment_domain = 'user'
            self.environment_name = environment_class.__name__

    def load_history(self, history_file):
        """
        Load benchmark history from file

        Args:
            history_file: path to history file

        Returns:

        """
        logging.info("Loading benchmark history data from {}".format(history_file))
        with open(os.path.join(os.getcwd(), history_file), "rb") as fp:
            self.history_data = pickle.load(fp)

    def load_model(self, model_file):
        """
        Load model from file

        Args:
            model_file: path to model file

        Returns:

        """
        logging.info("Loading model data from {}".format(model_file))
        self.load_model_file = model_file

    def episode_finished(self, runner):
        """
        Callback that is called from the runner after each finished episode. Outputs result summaries and saves history.

        Args:
            runner: TensorForce `Runner` object

        Returns: Boolean indicating whether to continue run or not.

        """
        if runner.episode % self.report_episodes == 0:
            logging.info("Finished episode {ep} after {ts} timesteps".format(ep=runner.episode, ts=runner.episode_timestep))
            logging.info("Episode reward: {}".format(runner.episode_rewards[-1]))
            logging.info("Average of last 500 rewards: {:.2f}".format(np.mean(runner.episode_rewards[-500:])))
            logging.info("Average of last 100 rewards: {:.2f}".format(np.mean(runner.episode_rewards[-100:])))

        if self.save_history_file and self.save_history_episodes > 0:
            if runner.episode % self.save_history_episodes == 0:
                logging.debug("Saving benchmark history to {}".format(self.save_history_file))
                history_data = dict(
                    episode=runner.episode,
                    timestep=runner.episode_timestep,
                    episode_rewards=copy(runner.episode_rewards),
                    episode_timesteps=copy(runner.episode_timesteps),
                    episode_end_times=copy(runner.episode_times)
                )

                with open(self.save_history_file, 'wb') as fp:
                    pickle.dump(history_data, fp)

        return True

    def run(self,
            experiments=1,
            report_episodes=10,
            save_history_file=None,
            save_history_episodes=0,
            save_model_file=None,
            save_model_episodes=0):

        self.report_episodes = report_episodes
        self.save_history_file = save_history_file
        self.save_history_episodes = save_history_episodes
        self.save_model_file = save_model_file
        self.save_model_episodes = save_model_episodes

        self.current_run_results = list()

        logging.info("Running benchmark with {:d} experiments".format(experiments))

        (environment_class, environment_args, environment_kwargs) = self.environment_callback

        for i in xrange(experiments):
            config = self.config.copy()

            environment = environment_class(*environment_args, **environment_kwargs)

            agent = agents[config.agent](states_spec=environment.states,
                                         actions_spec=environment.actions,
                                         network_spec=config.network,
                                         config=config)

            if i == 0 and self.history_data:
                logging.info("Attaching history data to runner")
                history_data = self.history_data
            else:
                history_data = None

            if i == 0 and self.load_model_file:
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

            logging.info("Starting experiment {}".format(i + 1))

            experiment_start_time = int(time.time())
            runner.run(episodes=config.episodes, max_episode_timesteps=config.max_timesteps,
                       episode_finished=self.episode_finished)
            experiment_end_time = int(time.time())

            logging.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

            experiment_data = dict(
                initial_reset_time=0,
                episode_rewards=runner.episode_rewards,
                episode_timesteps=runner.episode_timesteps,
                episode_end_times=runner.episode_times,

                metadata=dict(
                    agent=config.agent,
                    episodes=config.episodes,
                    max_timesteps=config.max_timesteps,
                    environment_domain=self.environment_domain,
                    environment_name=self.environment_name,
                    tensorforce_version=tensorforce_version,
                    tensorflow_version=tensorflow_version,
                    start_time=experiment_start_time,
                    end_time=experiment_end_time
                ),
                config=dict(self.config.items())  # convert original Configuration object into dict
            )

            self.current_run_results.append(experiment_data)

    def save_results(self, output_file, append=False, force=False):
        """
        Save results to file.

        Args:
            output_file: path to output file (relative to `self.output_folder` or absolute path)
            append: Boolean indicating whether to append data if output file exists
            force: Boolean indicating whether to overwrite data if output file exists (append has preference)

        Returns:

        """
        output_file_path = os.path.join(self.output_folder, output_file)

        benchmark_data = self.current_run_results
        if os.path.exists(output_file_path):
            if not append and not force:
                logging.error("Output file exists and should not be appended to or overwritten. Aborting.")
                return False
            if append:
                logging.info("Loading data from existing output file")
                with open(output_file_path, 'rb') as fp:
                    old_benchmark_data = pickle.load(fp)
                    benchmark_data = old_benchmark_data + self.current_run_results
            elif force:
                logging.warning("Overwriting existing benchmark file.")

        logging.info("Saving benchmark data to {}".format(output_file_path))
        with open(output_file_path, 'wb') as fp:
            pickle.dump(benchmark_data, fp)

