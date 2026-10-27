import copy
from pathlib import Path

import pytest
import yaml

from futsal.recipes import loadRecipe, validateRecipe


def testCurriculumRetainsCanonicalIdsAndSourceSemantics():
    recipe = loadRecipe(Path("configs/curriculum.yaml"))
    generic, roles, unified = recipe.data["tasks"]
    assert generic["classMap"]["goalkeeperTeamB"] == "player"
    assert roles["classMap"]["goalkeeperTeamB"] == "goalkeeper"
    assert unified["classes"] == [item["name"] for item in recipe.catalog.labels["canonicalClasses"]]
    assert unified["classMap"]["lowerCornerSideB"] == "lowerCornerSideB"
    assert "referee" not in generic["classMap"]
    assert len(recipe.evidence) == 3


@pytest.mark.parametrize("case", ["inventedClass", "unknownSource", "missingValidation",
                                   "unsafeTaskId", "partialFinalOntology", "invalidFractions"])
def testInvalidResearchContractsAreRejected(case):
    recipe = loadRecipe(Path("configs/curriculum.yaml"))
    data = copy.deepcopy(recipe.data)
    if case == "inventedClass":
        data["tasks"][0]["classMap"]["inventedTeam"] = "player"
    elif case == "unknownSource":
        data["tasks"][0]["trainingSources"] = ["missing"]
    elif case == "missingValidation":
        data["publicValidation"]["sources"] = []
    elif case == "unsafeTaskId":
        data["tasks"][0]["taskId"] = "../outside"
    elif case == "partialFinalOntology":
        data["tasks"][-1]["classes"] = ["ball"]
        data["tasks"][-1]["classMap"] = {"ball": "ball"}
    else:
        data["pointBoxes"]["widthFraction"] = float("nan")
    with pytest.raises(ValueError):
        validateRecipe(data, recipe.catalog)


def testDuplicateYamlFieldsFailAtTheBoundary(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text("schemaVersion: 1\nschemaVersion: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        loadRecipe(path)


def testRecipeConfigurationIsNotRewritten():
    path = Path("configs/curriculum.yaml")
    original = path.read_bytes()
    loadRecipe(path)
    assert path.read_bytes() == original
    assert yaml.safe_load(original)["tasks"][-1]["classes"] == "canonical"
