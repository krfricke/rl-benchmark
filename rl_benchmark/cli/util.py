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

import copy
import json
import logging
import os
import readline
import sys

from getpass import getpass


def load_config(config_file, silent=False, default_config=None, default_config_file=None):
    if default_config:
        config = copy.copy(default_config)
    else:
        config = dict()

    config_path = os.path.expanduser(config_file)

    if os.path.exists(config_path):
        with open(config_path, 'r') as fp:
            config.update(json.load(fp))
    elif not silent:
        if config_file == default_config_file:
            logging.warning('No config file found, using default config...')
        else:
            raise OSError('Config file not found: {}'.format(config_file))

    return config


class AutoCompleter(object):
    """
    Autocompleter for list completion
    """
    def __init__(self, options):
        self.options = sorted(options)
        self.matches = list()

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
