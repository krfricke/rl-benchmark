tensorforce-benchmark: Benchmarking for TensorForce 
===================================================

In this repository we provide scripts for creating and analyzing benchmarks
 of reinforcement learning algorithms created with the [TensorForce library](https://github.com/reinforceio/tensorforce).
 

Creating benchmarks
-------------------

You can easily create benchmarks using pre-supplied config files or your own configurations.

```bash
python benchmark.py [--output output] [--append] <algorithm> <gym_id>
```

`algorithm` specifies which config file to use. You can pass the path to a valid json config file, or a string
indicating which prepared config to use (e.g. `dqn2015`).

`gym_id` should be a valid [OpenAI gym ID](https://gym.openai.com/envs)

`output` is an optional parameter to set the output (pickle) file. If omitted, output will be saved in `./benchmarks`.

`append` is an optional parameter which indicates if data should be appended to an existing output file.

Analyzing benchmarks
--------------------

At the moment, we provide plotting of the results obtained from our benchmarking script.

```bash
python results.py [--output output] [--input <file> <name>] [--input <file> <name> ...]
```

`input` expects two parameters. `file` points to a pickle file (pkl) containing experiment data (e.g. created by
running `benchmark.py`). `name` is a string containing the label for the plot.

`output` is an optional parameter to set the output (png) file. If omitted, output will be saved as `./output.png`.

The resulting output file is a PNG image containing plots for rewards by episodes and rewards by timesteps.