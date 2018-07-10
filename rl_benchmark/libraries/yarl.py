# Copyright 2018 The YARL Project. All Rights Reserved.
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
YARL benchmarking.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow import __version__ as tensorflow_version

from yarl import __version__ as yarl_version

from rl_benchmark.benchmark.runner.benchmark_runner import BenchmarkRunner


class YarlBenchmarkRunner(BenchmarkRunner):
    rl_library = 'yarl'
    rl_library_version = yarl_version
    rl_backend = 'tensorflow'
    rl_backend_version = tensorflow_version

    def __init__(self, config=None, config_folder=None, output_folder='/tmp'):
        super(YarlBenchmarkRunner, self).__init__(config, config_folder, output_folder)

        self.environment_callback = None

        # This file is WIP
