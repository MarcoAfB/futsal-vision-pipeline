# Independent object and court detectors

The separated detection recipe defines detailedObjects (ball and four athlete
classes) and courtPoints (twenty detection classes). Each model starts independently
from the official YOLO26n checkpoint. Court landmarks retain the existing 1/64 by
1/36 boxes and native-point references. Only registered real own data are selected.

Projection recomputes eligibility from native annotations, keeping the canonical
unified25 task as the unchanged frozen split coverage basis. A landmark-only issue
does not exclude a valid object example, and an object-only issue does not exclude
a valid landmark example. Expected real train/validation counts are 2553/541 and
2579/548. The unsupported lowerCornerSideB validation score remains N/A. Test image
files remain reserved; training/test coverage is checked from eligibility metadata.
