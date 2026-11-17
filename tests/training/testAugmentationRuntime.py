import numpy as np
import pytest
from ultralytics.data.augment import Albumentations, Compose, RandomFlip
from ultralytics.utils.instance import Instances

from futsal.training.augmentationRuntime import (
    OriginGatedTransforms,
    SideSwapFlip,
    replaceHorizontalFlip,
    requireInactiveAlbumentations,
)


def sample():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, 0] = 255
    return {"img": image, "cls": np.array([[0.0], [2.0]], dtype=np.float32),
            "instances": Instances(np.array([[0.2, 0.5, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
                                   segments=np.zeros((0, 1000, 2), dtype=np.float32))}


def testSideSwapFlipMirrorsImageBoxesAndClasses():
    flipped = SideSwapFlip(1.0, [1, 0, 2])(sample())
    assert (flipped["img"][:, -1] == 255).all() and (flipped["img"][:, 0] == 0).all()
    assert flipped["instances"].bboxes[:, 0] == pytest.approx([0.8, 0.5])
    assert flipped["cls"].ravel().tolist() == [1.0, 2.0]


def testSideSwapFlipWithoutFlipKeepsSample():
    original = sample()
    kept = SideSwapFlip(0.0, [1, 0, 2])(sample())
    assert (kept["img"] == original["img"]).all()
    assert kept["cls"].ravel().tolist() == [0.0, 2.0]
    assert kept["instances"].bboxes[:, 0] == pytest.approx([0.2, 0.5])


def testOriginGateSendsExemptFilesToThePlainPipeline():
    gate = OriginGatedTransforms(lambda labels: "augmented", lambda labels: "plain", {"synthetic.png"})
    assert gate({"im_file": "synthetic.png"}) == "plain"
    assert gate({"im_file": "own.jpg"}) == "augmented"


def testHorizontalFlipIsReplacedOnce():
    transforms = Compose([RandomFlip(direction="vertical", p=0.0), RandomFlip(direction="horizontal", p=0.5)])
    replacement = SideSwapFlip(0.5, [0])
    replaceHorizontalFlip(transforms, replacement)
    assert transforms.transforms[1] is replacement and transforms.transforms[0].direction == "vertical"
    with pytest.raises(ValueError, match="exactly one native horizontal flip"):
        replaceHorizontalFlip(Compose([RandomFlip(direction="vertical", p=0.0)]), replacement)


def testActiveAlbumentationsIsRejected():
    step = Albumentations(p=1.0)
    requireInactiveAlbumentations(Compose([step]))
    step.transform = object()
    with pytest.raises(ValueError, match="undeclared training augmentations"):
        requireInactiveAlbumentations(Compose([step]))
