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
import hashlib
import json
import yaml
import logging
import os


def hash_object(obj):
    json_str = json.dumps(obj, sort_keys=True)
    hash_str = hashlib.sha1(json_str.encode('utf8')).hexdigest()
    return str(hash_str)


def load_config_file(filename, config_folder=None):
    possible_config_file_paths = [
        os.path.join(os.getcwd(), filename)  # first check absolute path
    ]

    if config_folder:
        possible_config_file_paths += [
            os.path.join(config_folder, '{}'.format(filename)),  # check with user-supplied file suffix, expects json
            os.path.join(config_folder, '{}.json'.format(filename)),  # check with json suffix
            os.path.join(config_folder, '{}.yml'.format(filename)),  # check with yml suffix
            os.path.join(config_folder, '{}.yaml'.format(filename))  # check with yaml suffix
        ]

    for possible_config_file_path in possible_config_file_paths:
        if not os.path.exists(possible_config_file_path):
            logging.debug("Possible config file does not exist: {}".format(possible_config_file_path))
            continue

        logging.debug("Found config file at {}".format(possible_config_file_path))
        with open(possible_config_file_path, 'r') as fp:
            if possible_config_file_path.endswith('yml') or possible_config_file_path.endswith('yaml'):
                return yaml.load(fp)
            else:
                return json.load(fp)

    return None


def n_step_average(data, n):
    """
    Average data over n steps.

    Args:
        data: np.array containing the ata
        n: steps to average over

    Returns: np.array of size n containing the average of the respective bins

    """
    if len(data) < n:
        n = len(data)

    cut = data[0:len(data)-len(data)%n]  # cut array so it's divisible by n
    return np.mean(cut.reshape(-1, len(data) // n), axis=1)
