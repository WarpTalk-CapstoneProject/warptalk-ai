import re
import tomllib
from pathlib import Path


def test_ingress_extra_includes_silero_runtime_imports() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    ingress_dependencies = project["project"]["optional-dependencies"]["ingress"]

    assert any(
        re.split(r"[<>=!~]", dependency, maxsplit=1)[0] == "packaging"
        for dependency in ingress_dependencies
    )
