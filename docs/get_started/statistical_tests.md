# Rliable statistical tests 📈

Stoix natively supports statistically robust evaluations using the Rliable library, ensuring that your reinforcement learning experiments are analyzed rigorously. This integration provides statistically confident comparisons between different algorithms.

#### What is Rliable?

Rliable is designed for robust evaluation and comparison of RL algorithms, emphasizing statistically sound methods for reporting performance.

#### Key Features

- **Interquartile Mean (IQM)**: Reduces the influence of outliers by focusing on the middle 50% of data.
- **Optimality Gap**: Measures the difference between obtained performance and optimal performance.
- **Bootstrapped Confidence Intervals**: Provides robust estimation of uncertainty around performance metrics.
- **Probability of Improvement**: Evaluates the likelihood that one algorithm outperforms another.
- **Performance Profiles**: Visualizes the proportion of tasks for which each algorithm is the best.

#### Configuration and Usage

Stoix integrates Rliable for evaluation and plotting, making it easy to generate plots and statistics directly from your training runs.

#### Usage

If you want to generate the Rliable plots, you must use the Stoix JSON logger to generate json files containing the evaluation performance.

After running your training scripts, you can use the plotting code available in [`plotting.ipynb`](https://github.com/EdanToledo/Stoix/blob/main/plotting.ipynb)

#### References and Further Reading

For more details, refer to the [Rliable documentation](https://github.com/google-research/rliable) as well as [MARL-EVAL](https://github.com/instadeepai/marl-eval) as we utilise their helper functions for plotting.
