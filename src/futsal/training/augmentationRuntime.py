from types import SimpleNamespace

import numpy as np
from ultralytics.data.augment import Albumentations, RandomFlip

from futsal.training.augmentationPolicy import (
    augmentationKeys,
    augmentationPolicies,
    mirroredClassIndices,
)


class SideSwapFlip(RandomFlip):
    """Horizontal flip that also relabels mirrored court landmarks (Side A <-> Side B)."""

    def __init__(self, p, classMap):
        super().__init__(p=p, direction="horizontal")
        self.classMap = np.asarray(classMap, dtype=np.float32)

    def apply_instances(self, labels, params):
        labels = super().apply_instances(labels, params)
        if params["flip"] and len(labels["cls"]):
            labels["cls"] = self.classMap[labels["cls"].astype(int)].reshape(labels["cls"].shape)
        return labels


class OriginGatedTransforms:
    def __init__(self, augmented, plain, plainFiles):
        self.augmented, self.plain, self.plainFiles = augmented, plain, frozenset(plainFiles)

    def __call__(self, labels):
        return (self.plain if labels["im_file"] in self.plainFiles else self.augmented)(labels)


def requireInactiveAlbumentations(transforms):
    if any(isinstance(item, Albumentations) and item.transform is not None for item in transforms.transforms):
        raise ValueError("Albumentations is installed and would apply undeclared training augmentations.")


def replaceHorizontalFlip(transforms, flip):
    positions = [index for index, item in enumerate(transforms.transforms)
                 if isinstance(item, RandomFlip) and item.direction == "horizontal"]
    if len(positions) != 1:
        raise ValueError("Expected exactly one native horizontal flip in the training pipeline.")
    transforms.transforms[positions[0]] = flip


def installTrainingTransforms(dataset, args, config, task, recordOf):
    requireInactiveAlbumentations(dataset.transforms)
    block = config.get("augmentation")
    if block is None:
        return {"policy": None}
    origins = {row["recordId"]: row["origin"] for row in task["training"]}
    exempt = {label["im_file"] for label in dataset.labels
              if origins[recordOf(label)] in block["exemptOrigins"]}
    plainArgs = SimpleNamespace(**{**vars(args), **dict.fromkeys(augmentationKeys, 0.0)})
    plain = dataset.build_transforms(hyp=plainArgs)
    classMap = mirroredClassIndices(task["classes"])
    replaceHorizontalFlip(dataset.transforms, SideSwapFlip(args.fliplr, classMap))
    dataset.transforms = OriginGatedTransforms(dataset.transforms, plain, exempt)
    names = task["classes"]
    return {"policy": block["policy"], "exemptOrigins": block["exemptOrigins"],
            "values": {key: getattr(args, key) for key in augmentationPolicies[block["policy"]]},
            "horizontalFlipClassMap": {names[index]: names[target] for index, target in enumerate(classMap)
                                       if index != target},
            "trainingRecords": len(dataset.labels), "exemptRecords": len(exempt)}
