"""Runtime registration for custom Ultralytics modules.

Ultralytics resolves YAML module names through ``ultralytics.nn.tasks`` globals
and its local ``parse_model`` channel rules. Custom modules that change channel
counts need both registrations.
"""

from __future__ import annotations

import inspect
import textwrap

from .c3k2_lfe import C3k2_LFE, LFE, LFELite
from .dysample import DySample
from .simam import SimAM
from .slimneck import GSConv, GSBottleneck, VoVGSCSP


def _inject_parse_model_rules(tasks_module) -> None:
    if getattr(tasks_module.parse_model, "_custom_improvements", False):
        return

    source = textwrap.dedent(inspect.getsource(tasks_module.parse_model))
    lines = source.splitlines()
    patched: list[str] = []
    a2c2f_count = 0

    for line in lines:
        patched.append(line)
        if line.strip() == "A2C2f,":
            a2c2f_count += 1
            indent = line[: len(line) - len(line.lstrip())]
            if a2c2f_count == 1:
                patched.extend([f"{indent}C3K2_LFE,", f"{indent}VoVGSCSP,"])
            elif a2c2f_count == 2:
                patched.extend([f"{indent}C3K2_LFE,", f"{indent}VoVGSCSP,"])

        if line.strip() == "elif m is AIFI:":
            indent = line[: len(line) - len(line.lstrip())]
            patched.pop()
            patched.extend(
                [
                    f"{indent}elif m in {{SimAM, LFE, LFELite}}:",
                    f"{indent}    c2 = ch[f]",
                    f"{indent}    args = [c2, *args[1:]] if args else [c2]",
                    f"{indent}elif m is DySample:",
                    f"{indent}    c2 = ch[f]",
                    f"{indent}    args = [c2, *args]",
                    line,
                ]
            )

    namespace = tasks_module.__dict__
    exec(compile("\n".join(patched), "<custom_ultralytics_parse_model>", "exec"), namespace)
    namespace["parse_model"]._custom_improvements = True


def register_improvements() -> None:
    """Register custom modules and patch Ultralytics YAML parsing."""
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    custom = {
        "C3K2_LFE": C3k2_LFE,
        "C3k2_LFE": C3k2_LFE,
        "DySample": DySample,
        "GSBottleneck": GSBottleneck,
        "GSConv": GSConv,
        "LFE": LFE,
        "LFELite": LFELite,
        "SimAM": SimAM,
        "VoVGSCSP": VoVGSCSP,
    }

    for name, obj in custom.items():
        setattr(modules, name, obj)
        setattr(tasks, name, obj)

    _inject_parse_model_rules(tasks)
