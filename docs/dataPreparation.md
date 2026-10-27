# Curriculum dataset preparation

The first experiment targets one detector with 25 classes: ball, four athlete
classes, and twenty fixed court landmarks. Data preparation follows three tasks:

| Task | Classes | Training sources |
|---|---|---|
| genericObjects | ball, player | Futsal4400, SmartFutsal |
| roleObjects | ball, outfieldPlayer, goalkeeper | FutsalMOT, FutsalAnaliza, FutsalDetect |
| unified25 | The ordered canonical ontology | Own training matches |

`configs/curriculum.yaml` declares task mappings separately from the ordered
curriculum. Original team and role labels remain available. A future experiment
can use a different task or sequence without changing the source ontology.
Training and checkpoint transfer are subsequent work; changing class indices
between stages does not establish semantic equivalence of classifier outputs.

`curriculum` selects an ordered subset of the declared tasks. A one-element
sequence prepares a standalone task with the same implementation. `coverageTask`
remains the complete canonical view used to validate the shared frozen split,
even when it is not exported. Inactive tasks receive no dataset directories,
public training samples or exported labels.

On 2026-10-26 the researcher confirmed manual validation of public annotations for
their source ontology and manual analysis of overlap with own recordings. This
declaration authorizes compatible public training use; no new automated overlap
search is performed. Preparation still checks mechanical format and geometry.
Public and synthetic data never contribute to the own-frame split denominator.
Synthetic court data is not used by this curriculum.

Whole original matches must remain together. All 25 classes require eligible
annotations in training and test after quarantine. Validation requires the other
24 classes; lowerCornerSideB is unsupported there. These constraints take priority
over approximate 70/15/15 proportions of the 3,684 reviewed own frame records.
Source records are not asserted to be unique decoded images.

## Task-specific quarantine

An invalid relevant annotation excludes the entire image from that task. Source
observations are retained, including the excluded targets. Geometry errors,
multiple balls, repeated semantic landmarks, and incompatible source shapes are
recorded as reasons. A landmark exported as an object box is not silently turned
into a point. Empty original public labels are quarantined. Known ignored classes
do not affect eligibility: invalid referee geometry does not exclude an otherwise
valid ball/player image. Generic athlete labels cannot acquire an invented role
or team. Structural source failures block planning rather than reducing the
authoritative reviewed frame denominator.

## Frozen original-match splits

Split selection enumerates original-match subsets. It first requires eligible
coverage of every canonical class in training and test, and exactly the declared
unsupported classes missing in validation. It then minimizes the sum of absolute
deviations from the target reviewed-frame counts. Ties prefer more eligible rare
class annotations in training, then lexicographically sorted training/validation/test
match groups. Integer-scaled target counts avoid floating-point tie differences.

The frozen configuration binds the assignment to a digest of own record IDs,
match grouping, task targets and eligibility. Changing eligibility requires an
explicit new split version rather than silently reusing or regenerating a split.
Reports show both the original reviewed-frame denominator and the retained own
frame denominator after task-specific quarantine.

The frozen first assignment in `configs/splits.yaml` has the following support:

| Split | Reviewed records | Fraction of 3,684 | Eligible unified25 records | Fraction of 3,626 |
|---|---:|---:|---:|---:|
| Training | 2,585 | 70.17% | 2,547 | 70.24% |
| Validation | 550 | 14.93% | 539 | 14.86% |
| Test | 549 | 14.90% | 540 | 14.89% |

Test contains match02, match09 and match14, with eligible annotations for all
25 classes. The rare lowerCornerSideB has 27 eligible test annotations and 61
training annotations. Both parts of match07 remain in training. These are
annotation eligibility counts; successful image decoding is checked during export.

## Detection geometry

Valid pixel boxes are normalized without clipping. Ball ellipses become their
axis-aligned enclosing boxes. A native landmark point becomes a centered box of
width 1/64 and height 1/36 of the image (30 by 30 pixels at 1920 by 1080).
Only artificial landmark boxes are intersected with the image boundary; the
transformation records whether truncation occurred. Original point coordinates
remain available for later localization evaluation, including at image edges
where a truncated box center differs from the original landmark.

Public boxes keep their original normalized coordinates. Labels use twelve decimal
places; a 1e-10 normalized-edge tolerance accommodates serialization rounding,
not invalid source geometry. Quarantined samples cannot enter conversion.

## Planning and source snapshots

```text
uv run python -m futsal planDataset --dataRoot ../Python --outputDir outputs/curriculumPlan
```

The default recipe is `configs/curriculum.yaml`; use `--config` for another recipe.
Its `splitPath` references the frozen split relative to the recipe. Omitting
`splitPath` selects a new candidate, written to the plan directory as `splits.yaml`;
freezing a new research split remains an explicit configuration change.

Planning writes `plan.json`, `splits.yaml` and `report.md` to a new directory.
The plan binds normalized recipes, raw annotation observations, eligibility,
split assignment, configuration hashes and SHA-256 hashes of source media,
annotations and evidence. It contains test annotation metadata but exports no
test images or YOLO labels. Reusing a changed source or plan is refused.
Hashes support reproducibility; they are not an automated overlap audit.

New directories are prepared beside their destination and renamed only after
successful completion. Existing destinations and source-root outputs are refused;
an interrupted operation does not publish a completed-looking directory.

## Development dataset export

```text
uv run python -m futsal prepareDataset --dataRoot ../Python --plan outputs/curriculumPlan/plan.json --outputDir data/prepared/curriculumV1
```

Each task has `dataset.yaml`, `images/train`, `images/val`, and matching label
directories. YAML paths are relative to the YAML directory and omit a test entry.
Own frames retain their original resolution and use lossless PNG with compression
level 1. Public images are copied byte-for-byte. Source paths and identifiers are
preserved in the manifest; derived filenames use SHA-256 of the record ID.

Videos are read sequentially, retrieving only selected development frames; frames
outside reviewed ranges never become samples. A decoder failure or dimension
mismatch stops publication. The manifest links exported rows to native annotations
in the embedded plan and records converted targets and transformations. Source
hashes are verified before and after export. The contract binds all output files.

`review.md` links annotated previews of applicable quarantined development samples.
Preview indices reference original annotations, and drawing clips are visualization
only. Reserved test samples remain exclusively in plan metadata and never receive
previews. Generic and role stages use own validation data, with public training only.

## Independent verification

```text
uv run python -m futsal verifyDataset --datasetDir data/prepared/curriculumV1
```

Preparation invokes this verifier before publishing the new directory. It can also
run without the source directory after moving the complete bundle to another
machine. It checks contract and artifact hashes, exact file membership, native
annotation references, converted labels, image decoding and dimensions, per-class
support, frozen splits, review membership and portable YAML paths. Extra test
entries, images or review previews are rejected. Updating file hashes alone does
not allow labels to disagree with the original annotation conversion.

Planning, preparation and verification return 0 on success and 2 on failure.
Expected quarantine does not produce a failed preparation. The existing inventory
command retains its separate 0/1/2 exit-code contract. Keep the prepared bundle
immutable and store future training outputs outside it. Test coverage here refers
to eligible annotation metadata; test images remain unexported and undecoded as
samples until a separate final-evaluation delivery.

## Verified local preparation

The Windows preparation completed using Python 3.13.5 and the project environment.
The installed module was invoked through `.venv/Scripts/python.exe -m futsal` for
the three commands above. Planning, export and standalone verification returned 0.

| Dataset | Exported training images | Exported validation images |
|---|---:|---:|
| genericObjects | 3,398 | 541 |
| roleObjects | 1,989 | 541 |
| unified25 | 2,547 | 539 |

These are 9,555 exported images across tasks, not 9,555 independent source images.
The bundle is `data/prepared/curriculumV1`, with 19,198 contracted artifact files
and 81 annotated quarantine previews. The 549 reserved test records remain in
metadata; 540 are eligible for unified25 and cover all 25 canonical classes.
The original 3,684-record denominator and retained proportions are reported above.

Public quarantine contains 26 multiple-ball samples, five originally empty labels
and one degenerate player box. The five invalid referee boxes do not remove images
from these tasks. Unified25 quarantines 58 own records across all splits, including
nine reserved test records. No original annotation was corrected or discarded.
Derived landmark boxes were intersected at image edges 290 times in unified25
training and 97 times in validation; their native coordinates remain in the plan.

SHA-256 checks matched for 10,864 source media, annotation and evidence files before
and after export. Standalone verification decoded all exported images, checked all
labels and artifact hashes, and confirmed task/split membership. An additional
sample from each of the eight development videos matched the original decoded
frame pixels exactly, including montage frame 180. This is a frame extraction
check, not certification of annotation correctness or athlete identity.

Plan ID: `8220bf1dcf3e2fc54dec8c15fd6a8e2ef8dc531903709f4961cf89ae3b6047e7`.

Contract ID: `3d7e7fe9ba0e45ee2bc4a0ac3f3a435ccd74c01285a26fa4eac6e595b4b1f0e1`.

Ruff passed and all 131 tests passed. The last suite used
`uv run --offline pytest --basetemp .pytest_cache/curriculumStandalone -q`.
Tests use generated fixtures and include a single active stage, portability,
modified labels and YAMLs, decoder interruption and source preservation.
No training, model inference or scientific performance evaluation was performed.
Backend compatibility and UTFPR GPU environment checks remain subsequent work.
