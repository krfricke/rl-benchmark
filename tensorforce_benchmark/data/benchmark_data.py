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

import numpy as np
import pickle

from tensorforce_benchmark.data import ExperimentData


class BenchmarkData(list):
    def __iter__(self):
        for item in super(BenchmarkData, self).__iter__():
            yield ExperimentData(item)

    def min_x(self, var):
        values = list()
        for experiment_data in self:
            values.append(np.max(experiment_data.to_timeseries()[var]))
        return np.min(values)

    @staticmethod
    def from_file(filename):
        """
        Load benchmark data from file.

        Args:
            filename: string of filename or file object

        Returns: BenchmarkData object

        """
        if isinstance(filename, file):
            return BenchmarkData(pickle.load(filename))
        else:
            with open(filename, 'rb') as fp:
                return BenchmarkData(pickle.load(fp))
