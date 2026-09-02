"""Load small pipeline modules without importing the GPU-heavy package API."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "pipeline"


def load_pipeline_module(name: str):
    """Load ``pipeline.<name>`` while bypassing ``pipeline.__init__``."""
    package = sys.modules.get("pipeline")
    if package is None:
        package = types.ModuleType("pipeline")
        package.__path__ = [str(PIPELINE_ROOT)]
        sys.modules["pipeline"] = package

    dotted_name = name.replace("/", ".")
    parts = dotted_name.split(".")
    for depth in range(1, len(parts)):
        parent_name = "pipeline." + ".".join(parts[:depth])
        if parent_name in sys.modules:
            continue
        parent = types.ModuleType(parent_name)
        parent.__path__ = [
            str(PIPELINE_ROOT.joinpath(*parts[:depth]))
        ]
        sys.modules[parent_name] = parent

    qualified_name = f"pipeline.{dotted_name}"
    existing = sys.modules.get(qualified_name)
    if existing is not None:
        return existing

    relative_name = name.replace(".", "/")
    spec = importlib.util.spec_from_file_location(
        qualified_name,
        PIPELINE_ROOT / f"{relative_name}.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {qualified_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Do not leave a partially initialized module that makes the next
        # test fail with a misleading missing-attribute error.
        sys.modules.pop(qualified_name, None)
        raise
    return module
