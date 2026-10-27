# Annotation formats

## CVAT recordings

The reader supports track and image exports with task or job metadata. Dimensions
can appear directly under `meta` or within the task/job record, as in the existing
exports. Conflicting declarations are reported rather than replaced.

Only explicitly exported shapes are read. Track frames are not interpolated, and
metadata describing video size or duration does not extend reviewed ranges.
Shapes marked `outside=1` are counted separately and excluded from active observations.

Coordinates retain their source representation:

| Shape | Values | Coordinate system |
|---|---|---|
| Box | left, top, right, bottom | Image pixels |
| Ellipse | centerX, centerY, radiusX, radiusY | Image pixels |
| Points | x, y pairs | Image pixels |

Nonfinite values and invalid bounds remain available for the subsequent audit.
Unsupported shapes or malformed numeric fields produce an explicit read failure.
Source labels, attributes, occlusion, and track IDs are retained. Track IDs are
annotation references and do not establish persistent athlete identity.

Image export IDs remain export indices. The associated filenames are recorded,
but the inventory requires an explicit mapping before treating them as video frames.

The audit reports nonfinite coordinates, nonpositive sizes, and geometry extending
beyond the image. It preserves the original values, including partially visible
objects. A finding requests review; it is not an instruction to clip or delete a label.

## YOLO image datasets

Each nonempty row has `classId centerX centerY width height`, with normalized box
coordinates. The same representation is used for synthetic landmark classes.
Class IDs use an explicit source mapping. Generic player roles retain their source
semantics without an invented team, and referees remain outside target counts.
Parsed observations remain counted when geometry needs review; malformed rows and
unknown IDs are reported with their annotation path and line number.

Preparation can also read individual YOLO observations, retaining their source
class ID, line number, normalized coordinates, mapping action and geometry
findings. This does not change the inventory JSON contract. Invalid coordinates
remain explicit rather than being silently corrected or dropped.
