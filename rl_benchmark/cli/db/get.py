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


class GetCommand(Command):
    def run(self, args):

        self.parser.add_argument('benchmark_hash', help="Benchmark hash to fetch")
        self.parser.add_argument('-f', '--force', action='store_true', default=False,
                                 help="Force request (don't use cached results)")
        self.parser.add_argument('-s', '--store-local', action='store_true', default=False, help="Store in local db")
        self.parser.add_argument('-o', '--output', help="Output filename")
        self.parser.add_argument('-j', '--json', action='store_true', default=False,
                                 help="Store in json format or output as json when --output is not given.")
        args = self.parser.parse_args(args)

        if not args.output and not args.store_local and not json:
            logging.error("Please tell me what to do with the benchmark: "
                          "--output to file, --store-local to local db, or print to stdout as --json")
            return 1

        if args.store_local and self.db == self.context['local_db']:
            logging.error("You're using the local database for lookup and want to store the result locally. "
                          "This will not work. Please consider querying the web database (usually with --web).")
            return 2

        benchmark = self.db.get_benchmark(args.benchmark_hash, force=args.force)

        if not benchmark:
            logging.error("Benchmark not found: {}".format(args.benchmark_hash))
            return 3

        if args.output:
            logging.debug("Saving benchmark to file {}".format(args.output))
            if args.output.endswith('.json') and not args.json:
                logging.warning("File suffix .json detected, assuming JSON format output.")
                args.json = True
            with open(args.output, 'w') as fp:
                if args.json:
                    json.dump(benchmark, fp)
                else:
                    pickle.dump(benchmark, fp)
        elif args.json:
            print(json.dumps(benchmark))

        if args.store_local:
            logging.debug("Saving benchmark to local db")
            local_db = self.context['local_db']
            local_db.save_benchmark(benchmark)

        return 0
