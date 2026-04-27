'''
Author: John McGuigan

Copyright: GPLv3 (see LICENSE file)

Cost-balanced candidate generation for hyperparameter search.

This module samples raw search-space configurations, estimates their relative
training cost from the Rhea e3nn model tier, batch size, and learning rate,
then preferentially selects lower-cost candidates for Syne Tune's
`points_to_evaluate`. It is used to seed schedulers with a more
compute-balanced initial trial set.
'''
from math import exp, log

import numpy as np

from .registry import CostBalancedSamplerCfg


class CostBalancedSampler:
    def __init__(self, cfg: CostBalancedSamplerCfg):
        self.cfg = cfg
        self.space = cfg.raw_space
        self.model_tiers = cfg.model_tiers
        self.budget_cfg = cfg.budget
        self.sampling_cfg = cfg.sampling
        self.base_model_cfg = cfg.base_model_cfg
        self.max_num_trials_started = cfg.max_num_trials_started
        self.random_state = np.random.RandomState(cfg.random_seed)

    @property
    def enabled(self) -> bool:
        return self.sampling_cfg is not None

    def num_points(self) -> int:
        if self.sampling_cfg.num_points is not None:
            return int(self.sampling_cfg.num_points)
        if self.max_num_trials_started is None:
            return self.candidate_pool_size()
        return int(self.max_num_trials_started)

    def candidate_pool_size(self) -> int:
        return int(self.sampling_cfg.candidate_pool_size)

    def cost_exponent(self) -> float:
        return float(self.sampling_cfg.cost_exponent)

    def sample_value(self, spec: dict):
        kind = spec["kind"]

        if kind == "constant":
            return spec["value"]

        if kind == "choice":
            values = spec["values"]
            return values[self.random_state.randint(0, len(values))]

        if kind == "loguniform":
            lower = log(float(spec["lower"]))
            upper = log(float(spec["upper"]))
            return exp(self.random_state.uniform(lower, upper))

        if kind == "uniform":
            return self.random_state.uniform(float(spec["lower"]), float(spec["upper"]))

        if kind == "randint":
            return int(self.random_state.randint(int(spec["lower"]), int(spec["upper"]) + 1))

        if kind == "lograndint":
            lower = log(float(spec["lower"]))
            upper = log(float(spec["upper"]))
            return int(round(exp(self.random_state.uniform(lower, upper))))

        return spec["value"]

    def sample_config(self) -> dict:
        return {
            name: self.sample_value(spec)
            for name, spec in self.space.items()
        }

    def estimate_params(self, config: dict) -> int:
        import e3nn.o3

        model_cfg = dict(self.base_model_cfg)
        tier = self.model_tiers[config["model_tier"]]
        model_cfg.update(tier.get("parms", {}))

        input_irreps = e3nn.o3.Irreps(model_cfg["input_irreps"])
        hidden_irreps = e3nn.o3.Irreps(model_cfg["irreps_hidden"])
        growthrate_irreps = e3nn.o3.Irreps(model_cfg["growthrate_irreps"])
        F4_irreps = e3nn.o3.Irreps(model_cfg["F4_irreps"])

        params = self.gated_block_params(
            input_irreps,
            hidden_irreps,
            model_cfg["tensor_product_class"],
        )
        params += (int(model_cfg["nhidden_shared"]) - 1) * self.gated_block_params(
            hidden_irreps,
            hidden_irreps,
            model_cfg["tensor_product_class"],
        )

        hidden_block_params = self.gated_block_params(
            hidden_irreps,
            hidden_irreps,
            model_cfg["tensor_product_class"],
        )
        params += int(model_cfg["nhidden_growthrate"]) * hidden_block_params
        params += self.linear_params(hidden_irreps, growthrate_irreps)
        params += int(model_cfg["nhidden_F4"]) * hidden_block_params
        params += self.linear_params(hidden_irreps, F4_irreps)

        return params

    @staticmethod
    def gated_block_params(irreps_in, irreps_out, tensor_product_class: str) -> int:
        import e3nn.o3

        irreps_scalars = irreps_out.filter(lambda mul_ir: mul_ir.ir.l == 0)
        irreps_nonscalars = irreps_out.filter(lambda mul_ir: mul_ir.ir.l > 0)
        irreps_gates = e3nn.o3.Irreps(f"{irreps_nonscalars.num_irreps}x0e")
        irreps_with_gates = irreps_scalars + irreps_gates + irreps_nonscalars

        if tensor_product_class == "norm":
            irreps_context = e3nn.o3.Norm(irreps_in).irreps_out
        elif tensor_product_class == "quadratic":
            irreps_context = irreps_in
        else:
            raise ValueError(f"Unknown tensor product class {tensor_product_class}")

        tensor_product = e3nn.o3.FullyConnectedTensorProduct(
            irreps_in,
            irreps_context,
            irreps_with_gates,
        )
        return 4 * int(tensor_product.weight_numel)

    @staticmethod
    def linear_params(irreps_in, irreps_out) -> int:
        import e3nn.o3

        layer = e3nn.o3.Linear(irreps_in, irreps_out)
        return int(sum(param.numel() for param in layer.parameters()))

    def estimate_cost(self, config: dict) -> float:
        param_scale = (
            float(self.estimate_params(config))
            / float(self.budget_cfg.ref_params)
        ) ** float(self.budget_cfg.param_exponent)
        batch_scale = (
            float(config["batch_size"])
            / float(self.budget_cfg.ref_batch_size)
        ) ** float(self.budget_cfg.batch_exponent)
        lr_need = (
            float(self.budget_cfg.ref_lr)
            / float(config["lr"])
        ) ** float(self.budget_cfg.lr_exponent)

        return param_scale * batch_scale * lr_need

    def points_to_evaluate(self) -> list[dict] | None:
        if not self.enabled:
            return None

        candidate_count = max(self.candidate_pool_size(), self.num_points())
        candidates = [self.sample_config() for _ in range(candidate_count)]

        costs = np.array(
            [max(self.estimate_cost(candidate), 1e-12) for candidate in candidates],
            dtype=float,
        )
        weights = costs ** (-self.cost_exponent())
        weights = weights / weights.sum()

        replace = self.num_points() > candidate_count
        indexes = self.random_state.choice(
            candidate_count,
            size=self.num_points(),
            replace=replace,
            p=weights,
        )

        return [candidates[int(index)] for index in indexes]
