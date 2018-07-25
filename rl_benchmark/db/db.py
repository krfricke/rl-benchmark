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
import os
import pickle

from rl_benchmark.data import BenchmarkData


class BenchmarkDatabase(object):
    def __init__(self):
        pass

    def load_config(self, config):
        raise NotImplementedError

    def load_config_file(self, config_file):
        with open(config_file, 'r') as fp:
            return self.load_config(json.load(fp))

    def get_benchmark(self, benchmark_hash):
        """
        Get benchmark from database.

        Args:
            benchmark_hash: benchmark_hash (unique benchmark identifier)

        Returns: `BenchmarkData` object

        """
        raise NotImplementedError

    def get_benchmark_info(self, benchmark_hash):
        """
        Get benchmark info from database.

        Args:
            benchmark_hash: benchmark_hash (unique benchmark identifier)

        Returns: dict

        """
        raise NotImplementedError

    def save_benchmark(self, benchmark_data):
        """
        Save benchmark to database.

        Args:
            benchmark_data: `BenchmarkData` object

        Returns: benchmark_hash

        """
        raise NotImplementedError

    def save_benchmark_file(self, benchmark_file):
        """
        Save benchmark to database.

        Args:
            benchmark_file: file path to benchmark data file.

        Returns: boolean

        """
        if not os.path.exists(benchmark_file):
            raise OSError("No such file: {}".format(benchmark_file))

        with open(benchmark_file, 'rb') as fp:
            benchmark_data = BenchmarkData.from_file(fp)

        return self.save_benchmark(benchmark_data)

    def search_by_config(self, config):
        """
        Search for benchmarks by config

        Args:
            config: Config dict or config_hash

        Returns:

        """
        raise NotImplementedError
