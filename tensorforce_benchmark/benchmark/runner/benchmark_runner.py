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

from collections import OrderedDict
from copy import copy
from six.moves import xrange
from tqdm import tqdm

from tensorflow import __version__ as tensorflow_version

from tensorforce import Configuration, __version__ as tensorforce_version
from tensorforce_benchmark.util import load_config_file


class BenchmarkRunner(object):
    def __init__(self, config=None, config_folder=None, output_folder='/tmp'):
        self.config = config
        self.config_folder = config_folder
        self.output_folder = output_folder

        self.history_data = None
        self.load_model_file = None

        self.save_history_file = None
        self.save_history_episodes = 0

        self.save_model_file = None
        self.save_model_episodes = 0

        self.current_run_results = None

        self.report_episodes = 10
        self.progress_bar = None

        self.environment_domain = 'user'
        self.environment_name = None

    def load_config(self, filename):
        """
        Load config from file. Either state a file from the config_folder (with or without file suffix),
        or supply full file path, or pass `tensorforce.config.Configuration` object.

        Args:
            filename: Filename, or full file path, or `tensorforce.config.Configuration` object

        Returns: Boolean

        """
        if isinstance(filename, Configuration):
            config = filename
        else:
            config = load_config_file(filename, config_folder=self.config_folder)

        if not config:
            return False

        if self.config:
            logging.warning("Overwriting existing configuration")

        self.config = config
        return True

    def set_environment(self, environment, *args, **kwargs):
        """
        Set environment and store as callback.

        Args:
            environment: Environment object
            *args: arguments to pass to environment class constructor
            **kwargs: keyword arguments to pass to environment class constructor

        Returns:

        """
        raise NotImplementedError

    def make_environment(self):
        """
        Create environment.

        Returns: environment

        """
        raise NotImplementedError

    def run_experiment(self, environment, experiment_num=0):
        """
        Learn.

        Args:
            environment: environment
            experiment_num: experiment number

        Returns:

        """
        raise NotImplementedError

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

    def episode_finished(self, results):
        """
        Callback that is called from the runner after each finished episode. Outputs result summaries and saves history.

        Args:
            results: results object (or TensorForce runner)

        Returns: Boolean indicating whether to continue run or not.

        """
        if self.progress_bar:
            self.progress_bar.update(1)
            self.progress_bar.set_postfix(OrderedDict([
                ('R', '{:8.0f}'.format(results.episode_rewards[-1])),
                ('AR100', '{:8.2f}'.format(np.mean(results.episode_rewards[-100:]))),
                ('AR500', '{:8.2f}'.format(np.mean(results.episode_rewards[-500:])))
            ]))
        else:
            if results.episode % self.report_episodes == 0:
                logging.info("Finished episode {ep} after {ts} timesteps".format(ep=results.episode, ts=results.episode_timestep))
                logging.info("Episode reward: {}".format(results.episode_rewards[-1]))
                logging.info("Average of last 500 rewards: {:.2f}".format(np.mean(results.episode_rewards[-500:])))
                logging.info("Average of last 100 rewards: {:.2f}".format(np.mean(results.episode_rewards[-100:])))

        if self.save_history_file and self.save_history_episodes > 0:
            if results.episode % self.save_history_episodes == 0:
                logging.debug("Saving benchmark history to {}".format(self.save_history_file))
                history_data = dict(
                    episode=results.episode,
                    timestep=results.episode_timestep,
                    episode_rewards=copy(results.episode_rewards),
                    episode_timesteps=copy(results.episode_timesteps),
                    episode_end_times=copy(results.episode_times)
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

        for i in xrange(experiments):
            config = self.config.copy()

            environment = self.make_environment()

            logging.info("Starting experiment {:d}".format(i + 1))

            with tqdm(total=config.max_episodes, desc='Experiment {:d}'.format(i + 1)) as self.progress_bar:
                experiment_start_time = int(time.time())
                results = self.run_experiment(environment, i)
                experiment_end_time = int(time.time())

            logging.info("Learning finished.")

            experiment_data = dict(
                results=results,
                metadata=dict(
                    agent=config.agent,
                    episodes=config.max_episodes,
                    max_timesteps=config.max_episode_timesteps,
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

        return self.current_run_results

    def save_results_db(self, db):
        """
        Save results to database.

        Args:
            db: `Database` object

        Returns: dict containing returned information on save status

        """
        benchmark_data = self.current_run_results
        return db.save_benchmark(benchmark_data)

    def save_results_file(self, output_file, append=False, force=False):
        """
        Save results to file.

        Args:
            output_file: path to output file (relative to `self.output_folder` or absolute path)
            append: Boolean indicating whether to append data if output file exists
            force: Boolean indicating whether to overwrite data if output file exists (append has preference)

        Returns: boolean

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

        return True