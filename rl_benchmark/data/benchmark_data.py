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

import numpy as np
import os
import pickle

from rl_benchmark.data import ExperimentData


class BenchmarkData(list):
    def __iter__(self):
        for item in super(BenchmarkData, self).__iter__():
            yield ExperimentData(item)

    def __getitem__(self, item):
        return ExperimentData(super(BenchmarkData, self).__getitem__(item))

    def min_x(self, var):
        values = list()
        for experiment_data in self:
            values.append(np.max(experiment_data.extended_results()[var]))
        return np.min(values)

    @staticmethod
    def from_file_or_hash(benchmark_lookup, db=None):
        """
        Load benchmark data from file or hash. First checks database(s) for hash, then files. Returns first match.

        Args:
            benchmark_lookup: string of filename, or file object, or local db hash
            db: `BenchmarkDatabase` object or list or `BenchmarkDatabase` objects

        Returns: BenchmarkData object

        """
        if isinstance(db, list):
            dbs = db
        else:
            dbs = [db]

        # Check for hash
        if isinstance(benchmark_lookup, str) and len(benchmark_lookup) == 40:
            for db in dbs:
                if not db:
                    continue
                benchmark_data = db.get_benchmark(benchmark_lookup)
                if benchmark_data:
                    return benchmark_data

        if hasattr(benchmark_lookup, 'readline') or os.path.exists(benchmark_lookup):
            return BenchmarkData.from_file(benchmark_lookup)
        else:
            raise ValueError("Could not find benchmark in db and fs: {}".format(benchmark_lookup))

    @staticmethod
    def from_file(filename):
        """
        Load benchmark data from file.

        Args:
            filename: string of filename or file object

        Returns: BenchmarkData object

        """
        if hasattr(filename, 'readline'):
            return BenchmarkData(pickle.load(filename))
        else:
            with open(filename, 'rb') as fp:
                return BenchmarkData(pickle.load(fp))
