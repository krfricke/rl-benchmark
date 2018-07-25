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

"""
Database cache class.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
import re

from distutils.dir_util import mkpath


def get_cache_file_name(identifier):
    return re.sub('[\W_]+', '_', identifier)


class Cache(object):
    """
    Database cache class to store get requests. Since we assume that individual benchmarks are immutable, there
    is no need to fetch them more than once from the database.
    """
    def __init__(self, cache_path='~/.cache/reinforce.io/general/'):
        self.cache_path = os.path.expanduser(cache_path)

    def _get_cache_file_path(self, identifier):
        """
        Return full cache file path.

        Args:
            identifier: object identifier (e.g. URL)

        Returns: full path

        """
        cache_file_name = get_cache_file_name(identifier)
        cache_file_path = os.path.join(self.cache_path, cache_file_name)

        return cache_file_path

    def get(self, identifier):
        """
        Get object from cache.

        Args:
            identifier: object identifier (e.g. URL)

        Returns: cached object

        """
        cache_file_path = self._get_cache_file_path(identifier)

        if os.path.isfile(cache_file_path):
            with open(cache_file_path, 'rb') as fp:
                result = pickle.load(fp)
            return result

        return None

    def save(self, data, identifier):
        """
        Save object to cache.

        Args:
            data: object to cache
            identifier: object identifier (e.g. URL)

        Returns: boolean

        """
        cache_file_path = self._get_cache_file_path(identifier)

        # Create path directory
        if not os.path.isdir(self.cache_path):
            logging.info("Creating cache directory at {}".format(self.cache_path))
            mkpath(self.cache_path, 0o755)

        with open(cache_file_path, 'wb') as fp:
            logging.debug("Storing result in cache file at {}".format(cache_file_path))
            pickle.dump(data, fp)

        return True
