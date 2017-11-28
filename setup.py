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
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires=[
    'numpy',
    'six',
    'scipy',
    'pillow',
    'pytest',
    'requests',
    'arghandler',
    'pyyaml',
    'tqdm>=4.11.0',
    'tensorforce>=0.3.2'
]

setup_requires=[
    'numpy',
    'recommonmark'
]

extras_require = {

}

setup(name='tensorforce_benchmark',
      version='0.0.3',  # please remember to edit tensorforce_benchmark/__init__.py when updating the version
      description='TensorForce benchmarking package',
      url='http://github.com/reinforceio/tensorforce-benchmark',
      author='reinforce.io',
      author_email='contact@reinforce.io',
      license='Apache 2.0',
      packages=[package for package in find_packages() if package.startswith('tensorforce_benchmark')],
      install_requires=install_requires,
      setup_requires=setup_requires,
      extras_require=extras_require,
      zip_safe=False)
