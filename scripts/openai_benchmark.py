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

import argparse
import logging
import os

from tensorforce_benchmark.benchmark.runner import BenchmarkRunner
from tensorforce.contrib.openai_gym import OpenAIGym


logging.basicConfig(level=logging.DEBUG)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('algorithm', help="Algorithm name (config file)")
    parser.add_argument('gym_id', help="ID of the gym environment")
    parser.add_argument('-o', '--output', help="output file (pickle pkl)")
    parser.add_argument('-x', '--experiments', default=1, type=int,
                        help="number of times to run the benchmark")
    parser.add_argument('-a', '--append', action='store_true', default=False,
                        help="Append data to existing pickle file?")
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help="Overwrite possible existing output file?")
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

    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

    benchmark_runner = BenchmarkRunner(
        config_folder=os.path.join(root, 'configs'),
        output_folder=os.path.join(root, 'benchmarks')
    )

    benchmark_runner.load_config(args.algorithm)

    # Set output file name
    if args.algorithm.endswith('.json'):
        output_filename = os.path.join(root, 'benchmarks', '{}_{}.pkl'.format(
            args.algorithm.replace('.', '_').replace('/', '__'), args.gym_id
        ))
    else:
        output_filename = os.path.join(root, 'benchmarks', '{}_{}.pkl'.format(
            args.algorithm, args.gym_id
        ))

    # Check if output file exists and should not be overwritten or appended to - we should not start the benchmark then
    if os.path.exists(os.path.join(benchmark_runner.output_folder, output_filename)) \
        and not args.force and not args.append:
        logging.error("Output file exists but should not be extended or overwritten, aborting.")
        return 1

    if args.load_model:
        benchmark_runner.load_model(args.load_model)

    if args.load_history:
        benchmark_runner.load_history(args.load_history)

    benchmark_runner.set_environment(OpenAIGym, args.gym_id)

    benchmark_runner.run(
        experiments=args.experiments,
        save_history_episodes=args.history_episodes,
        save_history_file=args.history,
        save_model_episodes=args.save_model,
        save_model_file=args.model
    )

    benchmark_runner.save_results(output_file=output_filename, append=args.append, force=args.force)

    return 0

if __name__ == '__main__':
    main()
