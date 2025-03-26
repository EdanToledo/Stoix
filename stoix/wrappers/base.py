from typing import Any, Dict, List

import hydra


def chained_wrappers(env: Any, wrapper_cfgs: List[Dict[str, Any]]) -> Any:
    """Apply multiple wrappers on top of each other"""

    def multi_wrapper(env: Any) -> Any:
        for wrapper_fn in [hydra.utils.instantiate(cfg, _partial_=True) for cfg in wrapper_cfgs]:
            env = wrapper_fn(env)

        return env

    return multi_wrapper(env)
