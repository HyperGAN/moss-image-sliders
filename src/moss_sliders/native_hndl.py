"""Opt-in MOSS spatial binding to the verified shared HyperGAN/HNDL graph.

The shared core owns the graph, GAN game, losses, optimizers and domain admission.
The product owns its frozen transformer, signed attention LoRA and velocity layout.
"""
from importlib.metadata import version
import hashlib
import json
from pathlib import Path

from torch import nn

PROFILE = "hypergan-hndl-v2"
LORA_GRAPH = '''down = linear(value, $rank, bias=False, name="down")
linear(down, $output_width, bias=False, name="up")'''


def verify_runtime():
    """Reject a historical environment before attaching or loading native weights."""
    if version("particle-sliders-core") != "0.12.0":
        raise ValueError("MOSS HNDL requires the separately qualified core 0.12.0 profile")
    import particle_sliders
    manifest_path = Path(__file__).with_name("hndl_cohort.json")
    cohort = json.loads(manifest_path.read_text())
    root = Path(particle_sliders.__file__).parent
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob("*") if p.suffix in (".py", ".json", ".hndl")}
    if actual != cohort["sources"]:
        raise ValueError("MOSS HNDL core source closure differs from the exact local candidate")
    from particle_sliders.pg09 import PG_VERSION, validate_pg09_domain
    from particle_sliders.fibo_hndl import verified_builder
    if PG_VERSION != "0.9.4":
        raise ValueError("MOSS HNDL requires official ParticleGAN 0.9.4")
    # Source validation is part of admission even for a tiny field.
    validate_pg09_domain((1, 4, 1, 2, 2), response_domain="continuous_field")
    _, identity = verified_builder()
    return dict(identity, core_commit=cohort["core_commit"],
                core_cohort_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                lora_graph_sha256=hashlib.sha256(LORA_GRAPH.encode()).hexdigest())


class HNDLLowRankResidual(nn.Module):
    """Actual HNDL attention LoRA with the historical native parameter names."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        from particle_sliders.fibo_hndl import verified_builder
        builder, identity = verified_builder()
        graph = builder(LORA_GRAPH, input_shape={"value": ["B", 1, base.in_features]},
                        output_shape=["B", 1, base.out_features],
                        parameters={"rank": rank, "output_width": base.out_features})
        graph.to(device=base.weight.device)
        self.down, self.up = graph["down"], graph["up"]
        nn.init.kaiming_uniform_(self.down.weight, a=1)
        nn.init.zeros_(self.up.weight)
        self.factor = alpha / rank
        self.input_width, self.output_width = base.in_features, base.out_features
        self.identity = identity
        # Register each parameter once under its existing portable LoRA name.
        # The graph references exactly these modules; no shadow leaves exist.
        object.__setattr__(self, "_graph", graph)

    def _apply(self, fn, recurse=True):
        # HNDL tracks runtime device/dtype alongside its registered leaves.
        # Convert through the graph once so those contracts move together;
        # down/up remain registered here under the portable adapter names.
        if recurse:
            self._graph._apply(fn, recurse=True)
            return self
        return super()._apply(fn, recurse=False)

    def train(self, mode=True):
        super().train(mode)
        self._graph.train(mode)
        return self

    def forward(self, value):
        shape = value.shape
        value = value.to(self.down.weight)
        result = self._graph(value=value.reshape(-1, 1, self.input_width))
        return result.reshape(*shape[:-1], self.output_width) * self.factor

    def extra_repr(self):
        return json.dumps(dict(self.identity,
            lora_graph_sha256=hashlib.sha256(LORA_GRAPH.encode()).hexdigest(),
            input_width=self.input_width, output_width=self.output_width,
            rank=self.down.out_features, factor=self.factor), sort_keys=True)


class MossHNDLFieldCritic(nn.Module):
    """Preserve MOSS spatial fields at the shared conv boundary."""
    score_bound = 0

    def __init__(self, response_shape, condition_width, width=32):
        super().__init__()
        from particle_sliders.pg09 import validate_pg09_domain
        from particle_sliders.fibo_hndl import HNDLFieldCritic
        shape = tuple(response_shape)
        validate_pg09_domain(shape, response_domain="continuous_field")
        if shape[0] != 1 or len(shape) not in (4, 5) or len(shape) == 5 and shape[2] != 1:
            raise ValueError("MOSS HNDL requires native [1,C,H,W] or [1,C,1,H,W] fields")
        self.identity = verify_runtime()
        self.response_shape = shape
        self.native = HNDLFieldCritic(shape[1], condition_width, shape[-2:], width)

    def forward(self, response, condition):
        if tuple(response.shape[1:]) != self.response_shape[1:]:
            raise ValueError("MOSS HNDL velocity geometry differs from its admitted native field")
        return self.native(response.squeeze(2) if response.ndim == 5 else response, condition)

    def extra_repr(self):
        return json.dumps(self.identity, sort_keys=True)


def build_hndl_run(pipe, contexts, config, **kwargs):
    """Construct an actual native adapter GAN using the HNDL critic graph."""
    from .gan import build_run
    return build_run(pipe, contexts, config, runtime_profile=PROFILE, **kwargs)


def main(argv=None):
    from .cli import main as cli_main
    return cli_main(argv, runtime_profile=PROFILE)
