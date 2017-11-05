tensorforce-benchmark: Benchmarking for TensorForce 
===================================================

In this repository we provide scripts for creating and analyzing benchmarks
 of reinforcement learning algorithms created with the [TensorForce library](https://github.com/reinforceio/tensorforce).
 

Creating benchmarks
-------------------

You can easily create benchmarks using pre-supplied config files or your own configurations. Per default, benchmarks
are stored in a local (sqlite) database.

```bash
python scripts/openai_benchmark.py [--output output] [--experiments num_experiments] [--append] [--model <path>] [--save-model <num_episodes>] [--load-model <path>] [--history <file>] [--history-episodes <num_episodes>] [--load-history <file>] <algorithm> <gym_id>
```

`algorithm` specifies which config file to use. You can pass the path to a valid json config file, or a string
indicating which prepared config to use (e.g. `dqn2015`).

`gym_id` should be a valid [OpenAI gym ID](https://gym.openai.com/envs)

`output` is an optional parameter to set the output (pickle) file. If omitted, output will be saved in `./benchmarks`.

`append` is an optional parameter which indicates if data should be appended to an existing output file.

`force` is an optional parameter which indicates if an existing output file should be overwritten.

`model` is an optional path for the `tf.train.Saver` class. If empty, model will not be saved.

`save-model <num_episodes>` states after how many episodes the model should be saved. If 0 or omitted,
model will not be saved.

`load-model <path>` states from which path to load the model (only for the first experiment, if more than one
experiment should run). If omitted, it does not load a model.

`history <file>` states the file where the history of the run should be periodically saved. If omitted, history will
not be saved.

`history-episodes <num_episodes>` states after how many episodes the history should be saved. If 0 or omitted,
history will not be saved.

`load-history <file>` states from which path to load the the run history (only for the first experiment, if more than one
experiment should run). If omitted, it does not load a history.


Analyzing benchmarks
--------------------

At the moment, we provide plotting of the results obtained from our benchmarking script.

```bash
python scripts/plot_results.py [--output output] [--show-episodes] [--show-timesteps] [--show-seconds] [--input <file> <name>] [--input <file> <name> ...]
```

`input` expects two parameters. `file` points to a pickle file (pkl) containing experiment data (e.g. created by
running `benchmark.py`). `name` is a string containing the label for the plot. You can state multiple input files.

`output` is an optional parameter to set the output image file. If omitted, output will be saved as `./output.png`.

`--show-*` indicates which values are to be used for the x axes.

The resulting output file is an image containing plots for rewards by episodes and rewards by timesteps.

This is a sample output for `CartPole-v0`, comparing VPG, TRPO and PPO (using the configurations provided in `configs`):

![example output](https://user-images.githubusercontent.com/14904111/30209005-328ea760-9496-11e7-93fc-80ea00794842.png)


Using Docker
------------

We provide a Docker image for benchmarking. The image currently only support creating benchmarks, not analyzing them.

Get started by pulling our docker image:

```bash
docker pull reinforceio/tensorforce-benchmark
```

Afterwards, you can start your benchmark. You should provide a host directory for the output files:

```bash
docker run -v /host/output:/benchmarks reinforceio/tensorforce-benchmark vpg_simple CartPole-v0
```

To provide your own configuration files, you can mount another host directory and pass the configuration file name as a parameter:

```bash
docker run -v /host/configs:/configs -v /host/output:/benchmarks reinforceio/tensorforce-benchmark my_config CartPole-v0
```

### Using tensorflow-gpu

We also provide a Docker image utilizing `tensorflow-gpu` on CUDA. You will need [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run this image.

First, pull the gpu image:

```bash
docker pull reinforceio/tensorforce-benchmark:latest-gpu
```

Then, run using `nvidia-docker`:

```bash
nvidia-docker run -v /host/configs:/configs -v /host/output:/benchmarks reinforceio/tensorforce-benchmark:latest-gpu my_config CartPole-v0
```

### Building Docker images

You can build the Docker images yourself using these commands:

```bash
# CPU version
docker build -f Dockerfile -t tensorforce-benchmark:latest .

# GPU version
nvidia-docker build -f Dockerfile.gpu -t tensorforce-benchmark:latest-gpu .
```

