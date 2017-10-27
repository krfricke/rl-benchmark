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

"""
TensorForce benchmarking.

Usage:

```bash
python benchmark.py [--output output] [--experiments num_experiments] [--append] [--model <path>] [--save-model <num_episodes>] [--load-model <path>] [--history <file>] [--history-episodes <num_episodes>] [--load-history <file>] <algorithm> <gym_id>
```

`algorithm` specifies which config file to use. You can pass the path to a valid json config file, or a string
indicating which prepared config to use (e.g. `dqn2015`).

`gym_id` should be a valid [OpenAI gym ID](https://gym.openai.com/envs)

`output` is an optional parameter to set the output (pickle) file. If omitted, output will be saved in `./benchmarks`.

`append` is an optional parameter which indicates if data should be appended to an existing output file.

`model` is an optional path for the `tf.train.Saver` class. If empty, model will not be saved.

`save-model <num_episodes>` states after how many episodes the model should be saved. If 0 or omitted,
model will not be saved.

`load-model <path>` states from which path to load the model (only for the first experiment, if more than one
experiment should run). If omitted, it does not load a model.

`history <file>` states the file where the history of the run should be periodically saved. If omitted, history will
not be saved.

`history-episodes <num_episodes>` states after how many episodes the history should be saved. If 0 or omitted,
history will not be saved.

`load-history <file>` states from which path to load the the run history (only for the first experiment, if more than one
experiment should run). If omitted, it does not load a history.

The resulting output file is a pickled python list, where each item is a dict containing benchmark data.

The dict has the following keys:

* `episode_rewards`: list containing observed total rewards for each episode.
* `episode_lengths`: list containing total timesteps for each episode.
* `initial_reset_time`: integer indicating starting timestamp (usually 0).
* `episode_end_times`: list containing observed end times relativ to `initial_reset_time` (not working at the moment).
* `info`: dict containing meta information about the experiment:
    * `agent`: TensorForce agent used in the experiment.
    * `episodes`: Episode count configuration item.
    * `max_timesteps`: Max timesteps configuration item.
    * `environment_name`: Environment name configuration item.
* `config`: `Configuration` object containing the original configuration passed to the benchmarking script.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pickle

from copy import copy

from six.moves import xrange

from tensorforce import Configuration
from tensorforce.agents import agents
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

logging.basicConfig()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('algorithm', help="Algorithm name (config file)")
    parser.add_argument('gym_id', help="ID of the gym environment")
    parser.add_argument('-o', '--output', help="output file (pickle pkl)")
    parser.add_argument('-x', '--experiments', default=1, type=int,
                        help="number of times to run the benchmark")
    parser.add_argument('-a', '--append', action='store_true', default=False,
                        help="Append data to existing pickle file?")
    parser.add_argument('-m', '--model', default=None, help="model path")
    parser.add_argument('-s', '--save-model', default=0, type=int, help="save model every n episodes")
    parser.add_argument('-l', '--load-model', default=None, help="load model from this file")

    parser.add_argument('-H', '--history', default=None, help="benchmark history file")
    parser.add_argument('-E', '--history-episodes', default=0, type=int,
                        help="save benchmark history every n episodes")
    parser.add_argument('-L', '--load-history', default=None, help="load benchmark history data from this file")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    root = os.path.dirname(os.path.realpath(__file__))

    config_file = os.path.join(root, args.algorithm)
    if not os.path.exists(config_file):
        config_file = os.path.join(root, 'configs', '{}.json'.format(args.algorithm))
        if not os.path.exists(config_file):
            raise ValueError("No configuration found: {}".format(args.algorithm))
        benchmark_file = os.path.join(root, 'benchmarks', '{}_{}.pkl'.format(
            args.algorithm, args.gym_id
        ))
    else:
        benchmark_file = os.path.join(root, 'benchmarks', '{}_{}.pkl'.format(
            args.algorithm.replace('.', '_').replace('/', '__'), args.gym_id
        ))

    if args.output:
        benchmark_file = args.output
    else:
        if not os.path.isdir(os.path.join(root, 'benchmarks')):
            os.mkdir(os.path.join(root, 'benchmarks'), 0o755)

    if os.path.isfile(benchmark_file):
        logger.debug("Found existing benchmark file: {}".format(benchmark_file))

        if not args.append:
            raise ValueError("Output file already exists (and --append was not passed), aborting.")

        logger.info("Loading data from output file to append to...")
        with open(benchmark_file, 'rb') as f:
            benchmark_data = pickle.load(f)
    else:
        benchmark_data = list()

    config = Configuration.from_json(config_file)

    original_config = config.copy()  # Until reinforceio/tensorforce#54 is fixed, use a copy of the config

    report_episodes = 1

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            logger.info("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))

        if args.history and args.history_episodes > 0:
            if r.episode % args.history_episodes == 0:
                logger.info("Saving benchmark history to {}".format(args.history))
                save_data = dict(
                    episode=r.episode,
                    timestep=r.timestep,
                    episode_rewards=copy(runner.episode_rewards),
                    episode_timesteps=copy(runner.episode_timesteps),
                    episode_end_times=copy(runner.episode_times)
                )

                pickle.dump(save_data, open(args.history, 'wb'))

        return True

    if args.load_history:
        logger.info("Loading benchmark history data from {}".format(args.load_history))
        with open(os.path.join(os.getcwd(), args.load_history), "rb") as f:
            history_data = pickle.load(f)
    else:
        history_data=None

    logger.info("Starting benchmark for agent {agent} and Environment '{env}'".format(agent=config.agent,
                                                                                      env=args.gym_id))
    logger.info("Results will be saved in {}".format(os.path.abspath(benchmark_file)))

    for i in xrange(args.experiments):
        config = original_config.copy()

        environment = OpenAIGym(args.gym_id)

        agent = agents[config.agent](states_spec=environment.states,
                                     actions_spec=environment.actions,
                                     network_spec=config.network,
                                     config=config)

        if i == 0 and args.load_model:
            logger.info("Loading model data from file: {}".format(args.load_model))
            agent.load_model(args.load_model)

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

        logger.info("Starting experiment {}".format(i+1))

        runner.run(config.episodes, config.max_timesteps, episode_finished=episode_finished)

        logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

        experiment_data = dict(
            episode_rewards=runner.episode_rewards,
            episode_lengths=runner.episode_timesteps,
            initial_reset_time=0,
            episode_end_times=runner.episode_times,
            info=dict(
                agent=config.agent,
                episodes=config.episodes,
                max_timesteps=config.max_timesteps,
                environment_name=args.gym_id
            ),
            config=original_config
        )

        benchmark_data.append(experiment_data)

        environment.close()

    logger.info("Saving benchmark to {}".format(benchmark_file))
    pickle.dump(benchmark_data, open(benchmark_file, 'wb'))
    logger.info("All done.")

if __name__ == '__main__':
    main()