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
RL benchmark result uploading and downloading.

Usage:

```bash
python db.py ...
```

"""

import argparse
import logging
import os
import sys

from rl_benchmark import default_config_file as DEFAULT_CONFIG_FILE
from rl_benchmark.db import LocalDatabase, WebDatabase
from rl_benchmark.cli.util import load_config
from rl_benchmark.cli.db import commands as db_commands

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-C', '--config-file', default=DEFAULT_CONFIG_FILE, help="config file")
    parser.add_argument('-l', '--local', action='store_true', default=False, help="use local db")
    parser.add_argument('-w', '--web', action='store_true', default=False, help="use web db")

    parser.add_argument('command')
    parser.add_argument('args', nargs='*')

    args = parser.parse_args()

    config = load_config(args.config_file, default_config_file=DEFAULT_CONFIG_FILE)

    if args.local:
        db_type = 'local'
    elif args.web:
        db_type = 'web'
    else:
        db_type = config.get('db', 'local')

    logging.info("Using {} database".format(db_type))

    local_db = LocalDatabase(**config)
    web_db = WebDatabase(**config)

    if db_type == 'local':
        db = local_db
    elif db_type == 'web':
        db = web_db
    else:
        logging.error("No such database type: {}".format(db_type))
        return 1

    db_cmd_obj = db_commands.get(args.command)
    if not db_cmd_obj:
        logging.error("No such db command: {}".format(args.command))
        return 2

    context=dict(
        config_file=args.config_file,
        local_db=local_db,
        web_db=web_db
    )

    db_cmd = db_cmd_obj(db, name=args.command, context=context)

    return db_cmd.run(args.args)

if __name__ == '__main__':
    sys.exit(main())
