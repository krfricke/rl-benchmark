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

import logging
import time
import numpy as np
import os
import pickle

from collections import OrderedDict
from copy import copy
from six.moves import xrange
from tqdm import tqdm

from rl_benchmark.util import load_config_file
from rl_benchmark.data import BenchmarkData


class BenchmarkRunner(object):
    rl_library = 'none'           # E.g. rlgraph
    rl_library_version = '0.0.0'  # E.g. rlgraph version
    rl_backend = 'none'           # E.g. tensorflow
    rl_backend_version = '0.0.0'  # E.g. tensorflow version

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
        self.limit_by_episodes = True  # if False, the run is limited by timesteps

        self.environment_domain = 'user'
        self.environment_name = None
        self.environment_callback = None

    def load_config(self, filename):
        """
        Load config from file. Either state a file from the config_folder (with or without file suffix),
        or supply full file path, or pass configuration dict object.

        Args:
            filename: Filename, or full file path, or dict object

        Returns: Boolean

        """
        if isinstance(filename, dict):
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
            environment: Environment object or name.
            *args: arguments to pass to environment class constructor
            **kwargs: keyword arguments to pass to environment class constructor

        Returns:

        """
        self.environment_callback = (environment, args, kwargs)

        if (isinstance(environment, str) and environment == 'openai_gym') or \
        environment.__name__ == 'OpenAIGym':
            self.environment_domain = 'openai_gym'
            self.environment_name = args[0]
        else:
            self.environment_domain = 'user'
            self.environment_name = environment.__name__

    def make_environment(self):
        """
        Create environment.

        Returns: environment

        """
        (environment_class, environment_args, environment_kwargs) = self.environment_callback

        if isinstance(environment_class, str):
            raise NotImplementedError("There is no general method for the creation of a string environment.")

        environment = environment_class(*environment_args, **environment_kwargs)

        return environment

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

    def episode_finished(self, results, runner_id):
        """
        Callback that is called from the runner after each finished episode. Outputs result summaries and saves history.

        Args:
            results: results object (or runner object)
            runner_id: runner id for distributed execution

        Returns: Boolean indicating whether to continue run or not.

        """
        if self.progress_bar:
            if self.limit_by_episodes:
                self.progress_bar.update(1)
            else:
                self.progress_bar.update(results.episode_timestep)
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

        self.current_run_results = BenchmarkData()

        max_episodes = self.config.get('max_episodes')
        max_timesteps = self.config.get('max_timesteps')

        assert bool(max_episodes) != bool(max_timesteps), 'Please limit either by episodes or by timesteps, not both'
        assert bool(max_episodes) or bool(max_timesteps), 'Please give a time limit for the run (episodes or timesteps)'

        if max_episodes:
            self.limit_by_episodes = True
            total = int(max_episodes)
        else:
            self.limit_by_episodes = False
            total = int(max_timesteps)

        logging.info("Running benchmark with {:d} experiments".format(experiments))

        for i in xrange(experiments):
            config = copy(self.config)

            environment = self.make_environment()

            logging.info("Starting experiment {:d}".format(i + 1))

            with tqdm(total=total, desc='Experiment {:d}'.format(i + 1)) as self.progress_bar:
                experiment_start_time = int(time.time())
                results = self.run_experiment(environment, i)
                experiment_end_time = int(time.time())

            logging.info("Learning finished.")

            experiment_data = dict(
                results=results,
                metadata=dict(
                    agent=config['type'],
                    episodes=max_episodes,
                    timesteps=max_timesteps,
                    max_episode_timesteps=config.get('max_episode_timesteps', 0),
                    environment_domain=self.environment_domain,
                    environment_name=self.environment_name,
                    rl_library=self.rl_library,
                    rl_library_version=self.rl_library_version,
                    rl_backend=self.rl_backend,
                    rl_backend_version=self.rl_backend_version,
                    start_time=experiment_start_time,
                    end_time=experiment_end_time
                ),
                config=dict(config)  # make sure this is a dict
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