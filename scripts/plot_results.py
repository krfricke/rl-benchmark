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
TensorForce benchmark result analysis and formatting.

Usage:

```bash
python plot_results.py [--output output] [--show-episodes] [--show-timesteps] [--show-seconds] [--input <file> <name>] [--input <file> <name> ...]
```

`input` expects two parameters. `file` points to a pickle file (pkl) containing experiment data (e.g. created by
running `benchmark.py`). `name` is a string containing the label for the plot. You can state multiple input files.

`output` is an optional parameter to set the output image file. If omitted, output will be saved as `./output.png`.

`--show-*` indicates which values are to be used for the x axes.

The resulting output file is a PNG image containing plots for rewards by episodes and rewards by timesteps.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import matplotlib.pyplot as plt
import os
import sys

from tensorforce_benchmark.analyze.plotter import ResultPlotter
from tensorforce_benchmark.data import BenchmarkData

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', action='append', nargs=2, metavar=('file', 'name'),
                        help="Input file(s): <file> <name>")
    parser.add_argument('-o', '--output', default="output.png", help="output file (image png)")

    parser.add_argument('-E', '--show-episodes', action='store_true', default=False,
                        help="show rewards by episode number")
    parser.add_argument('-T', '--show-timesteps', action='store_true', default=False,
                        help="show rewards by global timestep")
    parser.add_argument('-S', '--show-seconds', action='store_true', default=False,
                        help="show rewards by (wallclock) seconds")


    args = parser.parse_args()

    if len(args.input) < 1:
        raise ValueError("Please state at least one input file and name.")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    plotter = ResultPlotter()

    # load input files into data dict
    data = dict()
    for (filename, name) in args.input:
        path = os.path.join(os.getcwd(), filename)
        logger.info("Loading {} ({})".format(filename, name))

        benchmark_data = BenchmarkData.from_file(filename)
        plotter.add_benchmark(benchmark_data, name)

    num_plots = 0
    if args.show_episodes:
        num_plots += 1

    if args.show_timesteps:
        num_plots += 1

    if args.show_seconds:
        num_plots += 1

    if num_plots <= 0:
        logger.error("Please specify at least one plot type (-E, -T, or -S)")
        return

    max_row_length = 4
    if num_plots <= max_row_length:
        plot_rows = 1
        plot_cols = num_plots
    else:
        plot_rows = num_plots // max_row_length + 1
        plot_cols = max_row_length

    figure, axes = plt.subplots(ncols=plot_cols, nrows=plot_rows, figsize=(plot_cols * 6, plot_rows * 6))

    if num_plots == 1:
        axes = [axes]

    ax_index = -1

    if args.show_episodes:
        ax_index +=1
        plot = plotter.plot_reward_by_episode(axes[ax_index])
        figure.add_subplot(plot)

    if args.show_timesteps:
        ax_index +=1
        plot = plotter.plot_reward_by_timestep(axes[ax_index])
        figure.add_subplot(plot)

    if args.show_seconds:
        ax_index +=1
        plot = plotter.plot_reward_by_second(axes[ax_index])
        figure.add_subplot(plot)

    plt.tight_layout()

    logger.info("Saving figure to {}".format(args.output))
    figure.savefig(args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
