import math

from futsal.geometry.courtModel import landmarkNames

augmentationKeys = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
                    "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "cutmix",
                    "multi_scale", "close_mosaic")
# Upper bounds of the keys each policy may enable; every other augmentation key stays zero.
augmentationPolicies = {
    "courtSideSwapLight": {"hsv_h": 0.1, "hsv_s": 0.9, "hsv_v": 0.9, "translate": 0.3,
                           "scale": 0.5, "degrees": 10.0, "perspective": 0.001, "fliplr": 1.0},
}
exemptibleOrigins = {"synthetic"}


def validateAugmentation(block, stageIds):
    if not isinstance(block, dict) or set(block) != {"policy", "exemptOrigins"}:
        raise ValueError("augmentation requires exactly policy and exemptOrigins.")
    if block["policy"] not in augmentationPolicies:
        raise ValueError(f"Unknown augmentation policy: {block['policy']}.")
    origins = block["exemptOrigins"]
    if not isinstance(origins, list) or len(set(origins)) != len(origins) or set(origins) - exemptibleOrigins:
        raise ValueError(f"exemptOrigins must be distinct values from {sorted(exemptibleOrigins)}.")
    if stageIds != ["courtPoints"]:
        raise ValueError("Augmentation policies are supported only for independent courtPoints training.")


def validateAugmentationValues(parameters, policy):
    bounds = augmentationPolicies[policy] if policy else {}
    for key in augmentationKeys:
        value = parameters[key]
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError(f"Augmentation value {key} must be a finite number.")
        if key not in bounds and value != 0:
            message = (f"Augmentation {key} is not enabled by policy {policy}." if policy else
                       "All data augmentations must be disabled without an augmentation policy.")
            raise ValueError(message)
        if key in bounds and not 0 <= value <= bounds[key]:
            raise ValueError(f"Augmentation {key} must be within [0, {bounds[key]}] for policy {policy}.")


def mirroredName(name):
    if name not in landmarkNames:
        if "Side" in name or name.startswith("substitutionPoint"):
            raise ValueError(f"Unknown court landmark cannot be mirrored: {name}.")
        return name
    if name.endswith("SideA"):
        return name[:-1] + "B"
    if name.endswith("SideB"):
        return name[:-1] + "A"
    if name.startswith("substitutionPoint"):
        return f"substitutionPoint{5 - int(name[-1])}"
    return name


def mirroredClassIndices(names):
    """Class index map for a horizontal image mirror (court x -> length - x)."""
    positions = {name: index for index, name in enumerate(names)}
    mirrored = []
    for name in names:
        target = mirroredName(name)
        if target not in positions:
            raise ValueError(f"Mirrored class {target} of {name} is missing from the task.")
        mirrored.append(positions[target])
    return mirrored
