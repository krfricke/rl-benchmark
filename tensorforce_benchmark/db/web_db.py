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

import json
import logging
import requests
import urlparse

from tensorforce_benchmark.db import Cache
from tensorforce_benchmark.db.db import BenchmarkDatabase

API_VERSION = 'api/v1'


class WebDatabase(BenchmarkDatabase):
    def __init__(self,
                 webdb_url='https://benchmarks.reinforce.io',
                 webdb_cache='~/.cache/reinforce.io/webdb/',
                 auth_method='anonymous',
                 auth_credentials=None
                 ):

        super(WebDatabase, self).__init__()

        self.url = webdb_url
        self.cache = Cache(webdb_cache)
        self.auth_method = auth_method
        self.auth_credentials = auth_credentials

    def load_config_file(self, config_file):
        with open(config_file, 'r') as fp:
            return self.load_config(json.load(fp))

    def load_config(self, config):
        self.url = config.pop('url', self.url)

        cache = config.pop('cache', None)
        if cache:
            self.cache = Cache(cache)

        self.auth_method = config.pop('auth_method', self.auth_method)
        self.auth_credentials = config.pop('auth_credentials', self.auth_credentials)

        return True

    def get_benchmark(self, benchmark_hash, force=False):
        result = self.call_api('/benchmark/{}'.format(benchmark_hash), method='get', force=force)
        if result.status_code >= 400:
            return None
        return result.json()

    def get_benchmark_info(self, benchmark_hash, force=False):
        result = self.call_api('/benchmark/{}/info'.format(benchmark_hash), method='get', force=force)
        if result.status_code >= 400:
            return None
        return result.json()

    def save_benchmark(self, benchmark_data):
        result = self.call_api('/experiment', method='post', json=benchmark_data)
        if result.status_code >= 400:
            return False
        return True

    def search_by_config(self, config):
        raise NotImplementedError

    def call_api(self, endpoint, method='get', force=False, **kwargs):
        target_url = urlparse.urljoin(self.url, API_VERSION + endpoint)

        # Return cached get results
        if method == 'get' and not force:
            cached_result = self.cache.get(target_url)
            if cached_result:
                logging.debug("Returning cached result.")
                return cached_result

        headers = {
            'User-Agent': 'tensorforce-benchmark webdb',
            'Accept': 'application/json'
        }

        if self.auth_method != 'anonymous':
            if self.auth_method == 'userpw':
                real_auth_method = 'Basic'
            elif self.auth_method == 'apikey':
                real_auth_method = 'APIKEY'
            else:
                raise ValueError('No such auth method: {}'.format(self.auth_method))
            headers.update({'Authorization': '{} {}'.format(real_auth_method, self.auth_credentials)})

        result = requests.request(method, target_url, headers=headers, **kwargs)

        # Store cached result
        if method == 'get':
            logging.debug("Storing result in cache.")
            self.cache.save(result, target_url)

        return result
