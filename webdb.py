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

"""
TensorForce benchmark result uploading and downloading.

Usage:

```bash
python webdb.py [browse] [
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import pickle
import re
import readline
import requests
import urlparse
import sys

from arghandler import ArgumentHandler, subcmd
from base64 import b64encode
from distutils.dir_util import mkpath
from getpass import getpass

logging.basicConfig(level=logging.DEBUG)

API_VERSION = 'api/v1'

DEFAULT_CONFIG_FILE = os.path.expanduser('~/.config/reinforce_webdb.cfg')

WEBDB_CONFIG = dict(
    webdb_url='https://benchmarks.reinforce.io/',
    cache_path=os.path.expanduser('~/.cache/reinforce_webdb/'),
    auth_method='',
    auth_credentials=''
)


def get_cache_file_name(string):
    return re.sub('[\W_]+', '_', string)


def call_api(endpoint, method='get', config=None, force=False, **kwargs):
    webdb_config = WEBDB_CONFIG.copy()
    webdb_config.update(config)

    target_url = urlparse.urljoin(webdb_config['webdb_url'], API_VERSION + endpoint)

    # Return cached get results
    cache_path = webdb_config['cache_path']
    cache_file_name = get_cache_file_name(target_url)
    cache_file_path = os.path.join(cache_path, cache_file_name)

    if method == 'get' and not force:
        if os.path.isfile(cache_file_path):
            with open(cache_file_path, 'rb') as fp:
                logging.debug("Returning cached result from {}".format(cache_file_path))
                result = pickle.load(fp)
                return result

    target_url = urlparse.urljoin(webdb_config['webdb_url'], API_VERSION + endpoint)
    headers = {
        'User-Agent': 'tensorforce-benchmark webdb.py',
        'Accept': 'application/json'
    }

    if not webdb_config['auth_method'] == 'anonymous':
        headers.update({'Authorization': '{} {}'.format(webdb_config['auth_method'], webdb_config['auth_credentials'])})

    result = requests.request(method, target_url, headers=headers, **kwargs)

    # Store cached result
    if method == 'get':
        # Create path directory
        if not os.path.isdir(webdb_config['cache_path']):
            logging.info("Creating cache directory at {}".format(cache_path))
            mkpath(cache_path, 0o755)

        with open(cache_file_path, 'wb') as fp:
            logging.debug("Storing result in cache file at {}".format(cache_file_path))
            pickle.dump(result, fp)

    return result


class AutoCompleter(object):
    """
    Autocompleter for list completion
    """
    def __init__(self, options):
        self.options = sorted(options)

    def complete(self, text, state):
        if state == 0:
            if text:
                self.matches = [s for s in self.options if s and s.startswith(text)]
            else:
                self.matches = self.options[:]
        try:
            return self.matches[state]
        except IndexError:
            return None


def ask_yesno(label, default="yes"):
    valid = {"yes": True, "y": True, "Y": True,
             "no": False, "n": False, "N": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(label + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Invalid input\n")


def ask_string(label, default=None):
    string = None
    while not string:
        string = raw_input(label + ' ')
        if not string and default:
            return default
    return string


def ask_password(label, default=None):
    string = None
    while not string:
        string = getpass(label + ' ')
        if not string and default:
            return default
    return string


def ask_list(label, items, alt=None, default=None):
    completer = AutoCompleter(items)
    readline.set_completer(completer.complete)
    readline.set_completer_delims('')
    readline.parse_and_bind('tab: complete')

    item = None
    while not item:
        item = ask_string(label, default=default)
        if item not in items:
            if alt and item in alt:
                item = items[alt.index(item)]
            else:
                print("Invalid entry (try pressing TAB)")
                item = None

    readline.set_completer(None)
    return item


def load_config(config_file, silent=False):
    config = WEBDB_CONFIG.copy()

    if os.path.exists(config_file):
        with open(config_file, 'r') as fp:
            config.update(json.load(fp))
    elif not silent:
        if config_file == DEFAULT_CONFIG_FILE:
            logging.warning('No config file found, using default config...')
        else:
            raise OSError('Config file not found: {}'.format(config_file))

    return config


@subcmd('create-config')
def cmd_create_config(parser, context, args):
    print("This will create a config file at {path}".format(path=context.config_file))
    if not ask_yesno("Continue?", default='no'):
        return 1

    logging.info("Checking for existing config file...")
    config_file = os.path.join(os.getcwd(), context.config_file)
    config = load_config(config_file, silent=True)

    webdb_url = ask_string("Please state the benchmark db URI [{}]:".format(config['webdb_url']),
                           default=config['webdb_url'])
    auth_method = ask_list("Please select authentication method (anonymous|apikey|userpw) [{}]:".format(config['auth_method']),
                           items=['anonymous', 'apikey', 'userpw'], default=config['auth_method'])

    if auth_method == 'apikey':
        auth_method = 'APIKEY'  # for Authorization header
        auth_credentials = ask_password("Please paste your APIKEY (won't be printed):")
    elif auth_method == 'userpw':
        auth_method = 'Basic'  # for Authorization header
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
    print("Wrote configuration to {}".format(context.config_file))

    return 0


@subcmd('browse')
def cmd_browse(parser, context, args):
    config = load_config(context.config_file)

    print(config)
    parser.add_argument('action', help="Action: latest")
    args = parser.parse_args(args)

    if args.action == 'latest':
        print("Get latest benchmarks...")
    else:
        raise ValueError("No such action: {}".format(args.action))


@subcmd('upload')
def cmd_upload(parser, context, args):
    config_file = os.path.join(os.getcwd(), context.config_file)
    config = load_config(config_file)

    parser.add_argument('filename', help="File containing benchmark data")
    args = parser.parse_args(args)

    if not os.path.exists(args.filename):
        raise OSError("No such file: {}".format(args.filename))

    with open(args.filename, 'rb') as fp:
        benchmark_data = pickle.load(fp)

    result = call_api('/experiment', method='post', config=config, json=benchmark_data)
    print(result)
    print(result.text)


@subcmd('fetch')
def cmd_fetch(parser, context, args):
    config_file = os.path.join(os.getcwd(), context.config_file)
    config = load_config(config_file)

    parser.add_argument('benchmark_hash', help="Benchmark hash to fetch")
    parser.add_argument('-o', '--output', help="Output filename")
    parser.add_argument('-j', '--json', action='store_true', default=False, help="Store in json format")
    args = parser.parse_args(args)

    result = call_api('/benchmark/{}'.format(args.benchmark_hash), method='get', config=config,
                      force=context.force_reload)

    if args.output:
        if args.output.endswith('.json') and not args.json:
            logging.warning("File suffix .json detected, assuming JSON format output.")
            args.json = True
        with open(args.output, 'w') as fp:
            if args.json:
                json.dump(result.json(), fp)
            else:
                pickle.dump(result.json(), fp)
    else:
        print(result.json())

    return 0


@subcmd('info')
def cmd_fetch(parser, context, args):
    config_file = os.path.join(os.getcwd(), context.config_file)
    config = load_config(config_file)

    parser.add_argument('benchmark_hash', help="Benchmark hash to get info for")
    parser.add_argument('-o', '--output', help="Output filename (pkl or json)")
    parser.add_argument('-j', '--json', action='store_true', default=False, help="Print in json format")
    parser.add_argument('-c', '--print-config', action='store_true', default=False,
                        help="Print config to stdout (when not output file is given)")
    args = parser.parse_args(args)

    result = call_api('/benchmark/{}/info'.format(args.benchmark_hash), method='get', config=config,
                      force=context.force_reload)

    if args.output:
        if args.output.endswith('.json') and not args.json:
            logging.warning("File suffix .json detected, assuming JSON format output.")
            args.json = True
        with open(args.output, 'w') as fp:
            if args.json:
                json.dump(result.json(), fp)
            else:
                pickle.dump(result.json(), fp)
    else:
        if args.json:
            print(result.json())
        else:
            benchmark_data = result.json()
            if not benchmark_data:
                logging.error("No data received.")
                return 1

            benchmark_metadata = benchmark_data.get('metadata')
            if not isinstance(benchmark_metadata, dict):
                logging.warning("No metadata received.")
            else:

                # Pretty print
                print("Benchmark hash:\t{benchmark_hash}\n"
                      "--------------------------------\n"
                      "TensorFlow version:\t{tensorflow_version}\n"
                      "TensorForce version:\t{tensorforce_version}\n"
                      "\n"
                      "Environment domain:\t{environment_domain}\n"
                      "Environment name:\t{environment_name}\n"
                      "\n"
                      "Agent:\t\t\t{agent}\n"
                      "Episode count:\t\t{episodes}\n"
                      "Maximum timesteps/ep:\t{max_timesteps}\n"
                      "\n"
                      "Start time:\t\t{start_time}\n"
                      "End time:\t\t{end_time}\n".format(benchmark_hash=args.benchmark_hash, **benchmark_metadata))

            if args.print_config:
                benchmark_config = benchmark_data.get('config')
                if not isinstance(benchmark_config, dict):
                    logging.warning("No config received.")
                else:
                    print("Config hash: {config_hash}:\n"
                          "---------".format(config_hash=benchmark_data.get('config_hash')))
                    print(benchmark_config)

    return 0


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = ArgumentHandler()

    parser.add_argument('-C', '--config-file', default=DEFAULT_CONFIG_FILE, help="config file")
    parser.add_argument('-U', '--url', default=WEBDB_CONFIG['webdb_url'], help="webdb url")
    parser.add_argument('-F', '--force-reload', action='store_true', default=False,
                        help="force reload (don't use cached results)")

    parser.run()


if __name__ == '__main__':
    main()
