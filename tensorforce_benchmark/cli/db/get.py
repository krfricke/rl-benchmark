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

import json
import logging
import pickle

from tensorforce_benchmark.cli import Command


class GetCommand(Command):
    def run(self, args):

        self.parser.add_argument('benchmark_hash', help="Benchmark hash to fetch")
        self.parser.add_argument('-f', '--force', action='store_true', default=False,
                                 help="Force request (don't use cached results)")
        self.parser.add_argument('-o', '--output', help="Output filename")
        self.parser.add_argument('-j', '--json', action='store_true', default=False, help="Store in json format")
        args = self.parser.parse_args(args)

        result = self.db.get_benchmark(args.benchmark_hash, force=args.force)

        if not result:
            logging.error("Benchmark not found: {}".format(args.benchmark_hash))
            return 1

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
            print(result)

        return 0
