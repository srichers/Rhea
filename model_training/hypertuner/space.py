'''
Author: John McGuigan

Copyright: GPLv3 (see LICENSE file)

Search-space conversion helpers.

This module converts the project's serializable search-space schema into the
objects expected by `syne_tune.config_space`. Supported kinds include constant,
choice, uniform, loguniform, randint, and lograndint values.
'''
def build_config_space(space_cfg: dict) -> dict:
    from syne_tune.config_space import choice, lograndint, loguniform, randint, uniform

    config_space = {}

    for name, spec in space_cfg.items():
        kind = spec["kind"]

        if kind == "loguniform":
            config_space[name] = loguniform(spec["lower"], spec["upper"])
        elif kind == "uniform":
            config_space[name] = uniform(spec["lower"], spec["upper"])
        elif kind == "randint":
            config_space[name] = randint(spec["lower"], spec["upper"] + 1)
        elif kind == "lograndint":
            config_space[name] = lograndint(spec["lower"], spec["upper"] + 1)
        elif kind == "choice":
            config_space[name] = choice(spec["values"])
        else:
            config_space[name] = spec["value"]

    return config_space
