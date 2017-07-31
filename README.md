tensorforce-benchmark: Benchmarking for TensorForce 
===================================================

In this repository we provide scripts for creating and analyzing benchmarks
 of reinforcement learning algorithms created with the [TensorForce library](https://github.com/reinforceio/tensorforce).
 

Creating benchmarks
-------------------

```bash
python benchmark.py [--output output] [--append] <algorithm> <gym_id>
```

`algorithm` specifies which config file to use. You can pass the path to a valid json config file, or a string
indicating which prepared config to use (e.g. `dqn2015`).

`gym_id` should be a valid [OpenAI gym ID](https://gym.openai.com/envs)

`output` is an optional parameter to set the output (pickle) file. If omitted, output will be saved in `./benchmarks`.

`append` is an optional parameter which indicates if data should be appended to an existing output file.

