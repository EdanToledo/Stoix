from typing import NamedTuple

import chex


class ESEvaluation(NamedTuple):
    """Evaluation results from ES actor threads.

    Each actor evaluates perturbations of the base policy and reports
    the seeds used (for noise regeneration) and resulting fitnesses.
    """

    seeds: chex.Array  # [perturbations_per_actor] - int32 seeds for noise regeneration
    positive_fitnesses: chex.Array  # [perturbations_per_actor] - fitness of theta + sigma * noise
    negative_fitnesses: chex.Array  # [perturbations_per_actor] - fitness of theta - sigma * noise
