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

import json
import logging
import pickle

from rl_benchmark.cli import Command


class InfoCommand(Command):
    """
    Print info on benchmark.
    """
    def run(self, args):
        self.parser.add_argument('benchmark_hash', help="Benchmark hash to get info for")
        self.parser.add_argument('-f', '--force', action='store_true', default=False,
                                 help="Force request (don't use cached results)")
        self.parser.add_argument('-o', '--output', help="Output filename (pkl or json)")
        self.parser.add_argument('-j', '--json', action='store_true', default=False, help="Print in json format")
        self.parser.add_argument('-c', '--print-config', action='store_true', default=False,
                            help="Print config to stdout (when not output file is given)")
        args = self.parser.parse_args(args)

        result = self.db.get_benchmark_info(args.benchmark_hash, force=args.force)

        if args.output:
            if args.output.endswith('.json') and not args.json:
                logging.warning("File suffix .json detected, assuming JSON format output.")
                args.json = True
            with open(args.output, 'w') as fp:
                if args.json:
                    json.dump(result, fp)
                else:
                    pickle.dump(result, fp)
        else:
            if args.json:
                print(json.dumps(result))
            else:
                benchmark_data = result
                if not benchmark_data:
                    logging.error("No data received.")
                    return 1

                benchmark_metadata = benchmark_data.get('metadata')
                if not isinstance(benchmark_metadata, dict):
                    logging.warning("No metadata received.")
                else:

                    # Pretty print
                    print("Benchmark hash:\t{benchmark_hash}\n"
                          "--------------------------------\n"
                          "Library:\t{rl_library} ({rl_library_version})\n"
                          "Backend:\t{rl_backend} ({rl_backend_version})\n"
                          "\n"
                          "Environment domain:\t{environment_domain}\n"
                          "Environment name:\t{environment_name}\n"
                          "\n"
                          "Agent:\t\t\t{agent}\n"
                          "Episode count:\t\t{episodes}\n"
                          "Maximum timesteps/ep:\t{max_timesteps}\n"
                          "\n"
                          "Start time:\t\t{start_time}\n"
                          "End time:\t\t{end_time}\n".format(benchmark_hash=args.benchmark_hash, **benchmark_metadata))

                if args.print_config:
                    benchmark_config = benchmark_data.get('config')
                    if not isinstance(benchmark_config, dict):
                        logging.warning("No config received.")
                    else:
                        print("Config hash: {config_hash}:\n"
                              "---------".format(config_hash=benchmark_data.get('config_hash')))
                        print(benchmark_config)

        return 0