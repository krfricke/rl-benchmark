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
import sqlite3

from distutils.dir_util import mkpath

from rl_benchmark.db.db import BenchmarkDatabase
from rl_benchmark.data import ExperimentData, BenchmarkData

def result_to_experiment(result):
    """
    Convert (SQL) result in to `ExperimentData` object
    Args:
        result:

    Returns: `ExperimentData` object

    """
    experiment_hash, benchmark_hash, config_hash, metadata_txt, config_txt, results_txt = result
    experiment = ExperimentData(dict(
        metadata=json.loads(metadata_txt),
        config=json.loads(config_txt),
        results=json.loads(results_txt)
    ))

    return experiment


class LocalDatabase(BenchmarkDatabase):
    def __init__(self,
                 localdb_path='~/.rf_localdb/benchmarks.db',
                 *args,
                 **kwargs
                 ):

        super(LocalDatabase, self).__init__()

        self.path = os.path.expanduser(localdb_path)

        self.db_conn = None
        self.db_cursor = None

        self.init_db()

    def load_config(self, config):
        self.path = config.pop('localdb_path', self.path)
        self.init_db()

    def get_experiment(self, experiment_hash, force=True):
        conn, cursor = self.connect_db()

        vars = (experiment_hash,)
        cursor.execute("SELECT experiment_hash, benchmark_hash, config_hash, metadata, config, results "
                       " FROM experiments WHERE experiment_hash=?", vars)
        result = cursor.fetchone()

        self.close_db()

        if not result:
            logging.debug("Did not find experiment_hash {} in local db.".format(experiment_hash))
            return None

        return result_to_experiment(result)

    def get_benchmark(self, benchmark_hash, force=True):
        conn, cursor = self.connect_db()

        vars = (benchmark_hash,)
        cursor.execute("SELECT experiment_hash, benchmark_hash, config_hash, metadata, config, results "
                       " FROM experiments WHERE benchmark_hash=?", vars)
        results = cursor.fetchall()

        self.close_db()

        if len(results) == 0:
            logging.debug("Did not find benchmark_hash {} in local db.".format(benchmark_hash))
            return None

        benchmark_data = BenchmarkData([result_to_experiment(result) for result in results])

        return benchmark_data

    def get_benchmark_info(self, benchmark_hash, force=True):
        conn, cursor = self.connect_db()

        vars = (benchmark_hash,)
        cursor.execute("SELECT experiment_hash, benchmark_hash, config_hash, metadata, config, results "
                       " FROM experiments WHERE benchmark_hash=?", vars)
        results = cursor.fetchall()

        self.close_db()

        if len(results) == 0:
            logging.debug("Did not find benchmark_hash {} in local db.".format(benchmark_hash))
            return None

        experiment_hash, benchmark_hash, config_hash, metadata_txt, config_txt, results_txt = results[0]

        return dict(config_hash=config_hash, metadata=json.loads(metadata_txt), config=json.loads(config_txt))

    def save_benchmark(self, benchmark_data):
        if not isinstance(benchmark_data, BenchmarkData):
            benchmark_data = BenchmarkData(benchmark_data)

        added_experiments = list()  # list of experiment ids that have been added to the database
        added_experiment_hashes = list()  # list of experiment hashes that have been added to the database
        benchmark_hashes = list()  # list of benchmark hashes (both added and duplicates)
        duplicate_experiments = list()  # list of experiment ids that already were in the database
        duplicate_experiment_hashes = list()  # list of experiment hashes that already were in the database

        vars = list()
        for i, experiment_data in enumerate(benchmark_data):
            experiment_hash, benchmark_hash, config_hash = experiment_data.hash()

            benchmark_hashes.append(benchmark_hash)

            # check if experiment already exists
            if (self.get_experiment(experiment_hash)):
                logging.warning("Experiment with hash {} already exists, ignoring.".format(experiment_hash))
                duplicate_experiments.append(i)
                duplicate_experiment_hashes.append(experiment_hash)
                continue

            added_experiments.append(i)
            added_experiment_hashes.append(experiment_hash)

            config = experiment_data.get('config', dict())
            metadata = experiment_data.get('metadata', dict())
            results = experiment_data.get('results', dict())

            vars.append((
                experiment_hash,
                config_hash,
                benchmark_hash,
                metadata.get('agent'),
                metadata.get('max_episodes'),
                metadata.get('max_timesteps'),
                metadata.get('max_episode_timesteps'),
                metadata.get('environment_domain'),
                metadata.get('environment_name'),
                metadata.get('rl_library'),
                metadata.get('rl_library_version'),
                metadata.get('rl_backend'),
                metadata.get('rl_backend_version'),
                metadata.get('start_time', 0),
                metadata.get('end_time', 0),
                json.dumps(metadata, sort_keys=True),
                json.dumps(config, sort_keys=True),
                json.dumps(results, sort_keys=True)
            ))

        if len(vars) > 0:
            conn, cursor = self.connect_db()

            cursor.executemany("INSERT INTO experiments (experiment_hash, config_hash, benchmark_hash, "
                               "md_agent, md_max_episodes, md_max_timesteps, md_max_episode_timesteps, "
                               "md_environment_domain, md_environment_name, "
                               "md_rl_library, md_rl_library_version, md_rl_backend, md_rl_backend_version, "
                               "start_time, end_time, metadata, config, results) VALUES "
                               "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", vars)
            conn.commit()

            self.close_db()
        else:
            logging.warning("No rows inserted.")

        return dict(
            added_experiments=added_experiments,
            added_experiment_hashes=added_experiment_hashes,
            benchmark_hashes=benchmark_hashes,
            duplicate_experiments=duplicate_experiments,
            duplicate_experiment_hashes=duplicate_experiment_hashes
        )

    def search_by_config(self, config):
        pass

    def connect_db(self):
        if not self.db_conn or not self.db_cursor:
            self.db_conn = sqlite3.connect(self.path)
            self.db_cursor = self.db_conn.cursor()

        return self.db_conn, self.db_cursor

    def close_db(self):
        self.db_conn.close()
        self.db_conn = None
        self.db_cursor = None

        return True

    def init_db(self):
        if os.path.exists(self.path):
            return False

        logging.info("Creating local database at {}".format(self.path))

        dir = os.path.dirname(self.path)
        if not os.path.exists(dir):
            mkpath(dir, 0o755)

        conn, cursor = self.connect_db()

        cursor.execute("CREATE TABLE experiments (experiment_hash text, config_hash text, benchmark_hash text, "
                       "md_agent text, md_max_episodes integer, md_max_timesteps integer, "
                       "md_max_episode_timesteps integer, md_environment_domain text, "
                       "md_environment_name text, md_rl_library text, md_rl_library_version text, "
                       "md_rl_backend text, md_rl_backend_version text, start_time integer, end_time integer, "
                       "metadata text, config text, results text)")

        conn.commit()

        os.chmod(self.path, 0o600)

        self.close_db()


