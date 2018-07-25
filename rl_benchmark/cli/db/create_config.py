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
import os

from base64 import b64encode
from distutils.dir_util import mkpath

from rl_benchmark.cli.util import load_config, ask_yesno, ask_string, ask_password, ask_list
from rl_benchmark.cli import Command

DEFAULT_CONFIG = {
    'db': 'local',
    'localdb_path': '~/.rf_localdb/benchmarks.db',
    'webdb_url': 'https://benchmarks.reinforce.io',
    'auth_method': 'anonymous',
    'auth_credentials': None
}


class CreateConfigCommand(Command):
    def run(self, args):
        print("This will create a config file at {path}".format(path=self.context['config_file']))
        if not ask_yesno("Continue?", default='no'):
            return 1

        logging.info("Checking for existing config file...")
        config_file = os.path.join(os.getcwd(), self.context['config_file'])
        config = load_config(config_file, default_config=DEFAULT_CONFIG, silent=True)

        webdb_url = ask_string("Please state the benchmark db URI [{}]:".format(config['webdb_url']),
                               default=config['webdb_url'])
        auth_method = ask_list(
            "Please select authentication method (anonymous|apikey|userpw) [{}]:".format(config['auth_method']),
            items=['anonymous', 'apikey', 'userpw'], default=config['auth_method'])

        if auth_method == 'apikey':
            auth_credentials = ask_password("Please paste your APIKEY (won't be printed):")
        elif auth_method == 'userpw':
            auth_credentials_user = ask_string("Please insert your username:")
            auth_credentials_pw = ask_password("Please insert your password:")
            auth_credentials = b64encode("{username}:{password}".format(username=auth_credentials_user,
                                                                        password=auth_credentials_pw))
        else:
            auth_credentials = None

        config.update(dict(
            webdb_url=webdb_url,
            auth_method=auth_method,
            auth_credentials=auth_credentials
        ))

        if not os.path.exists(config_file):
            # Check if directory exists
            config_dir = os.path.dirname(config_file)
            if not os.path.isdir(config_dir):
                if not ask_yesno("Directory {} does not exist. Create?".format(config_dir), default='yes'):
                    print("Aborted.")
                    return 2
                mkpath(config_dir, 0o755)
                # else directory exists, but file doesn't

        with open(config_file, 'w') as fp:
            json.dump(config, fp, sort_keys=True)
        os.chmod(config_file, 0o600)
        print("Wrote configuration to {}".format(self.context['config_file']))

        return 0

