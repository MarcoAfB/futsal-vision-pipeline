from futsal.evaluation.modelComparison import modelChanges
from futsal.reporting.researchReport import markdownTable


def modelComparisonMarkdown(comparisonMode, models):
    lines = [f"Comparison mode: **{comparisonMode}**.", ""]
    rows = [[side, item["componentId"], f"yolo26{item['modelSize']}",
             item["initialWeightsSha256"], item["checkpointSha256"]]
            for side, components in models.items() for item in components]
    lines += markdownTable(["Side", "Component", "Model", "Initial SHA-256", "Checkpoint SHA-256"], rows)
    if modelChanges(models):
        lines += ["Model sizes and corresponding official initializations differ. These results compare complete "
                  "variants and do not isolate the effect of curriculum or synthetic supplementation.", ""]
    return lines
