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
OpenAI gym execution
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pickle

from tensorforce import Configuration
from tensorforce.agents import agents
from tensorforce.core.networks import layered_network_builder
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('algorithm', help="Algorithm name (config file)")
    parser.add_argument('gym_id', help="ID of the gym environment")
    parser.add_argument('-o', '--output', help="output file (pickle pkl)")

    args = parser.parse_args()

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

    if not os.path.isdir(os.path.join(root, 'benchmarks')):
        os.mkdir(os.path.join(root, 'benchmarks'), 0o755)

    config = Configuration.from_json(config_file)

    config.network = layered_network_builder(config.network)
    environment = OpenAIGym(args.gym_id)

    config.default(dict(states=environment.states, actions=environment.actions))
    agent = agents[config.agent](config=config)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    report_episodes = 1

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            logger.info("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    logger.info("Starting benchmark for agent {agent} and Environment '{env}'".format(agent=agent, env=environment))
    logger.info("Results will be saved in {}".format(os.path.abspath(benchmark_file)))

    runner.run(config.episodes, config.max_timesteps, episode_finished=episode_finished)

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

    benchmark_data = dict(
        episode_rewards=runner.episode_rewards,
        episode_lengths=runner.episode_lengths,
        initial_reset_time=0,
        episode_end_times=list()
    )

    environment.close()

    logger.info("Saving benchmark data of {} episodes to {}".format(len(runner.episode_rewards), benchmark_file))
    pickle.dump(benchmark_data, open(benchmark_file, 'wb'))
    logger.info("All done.")

if __name__ == '__main__':
    main()
