# Data sources and label semantics

The source catalog registers inputs for a read-only inventory. It does not assign
splits, convert annotations, extract frames, or modify source files.

## Paths and catalog contract

The inventory receives an explicit `--dataRoot data/raw` pointing to the local
raw source tree. Video, annotation, image, label and source metadata paths in
`configs/dataSources.yaml` are relative to that root.
`labelMapPath` is the exception: it is relative to the catalog file itself.
No machine-specific absolute path is stored in the catalog.

The source catalog uses `schemaVersion: 2`; the label map uses version1. The `expected` section describes reviewed
own frame records, not unique decoded images or annotation counts. Public and
synthetic images are excluded from these expected totals. Original video filenames,
XML labels, track IDs, and YOLO class IDs must remain available in inventory records.

Reviewed intervals in the project catalog are authoritative for all nine videos.
Extra frames in the XML or media are not eligible merely because they exist.
The montage uses the explicitly confirmed original-match intervals. No external
review registry or code/configuration from another project is needed.

`data/raw/sourceManifest.json` seals the10862 raw files and their hashes. Native
videos/XMLs are under own/videos and own/annotations; the five public sources are
under public/sourceId. Public record namespaces keep sample identity independent
of the storage directory. Annotation mapping evidence is the project's own label
map, already included in configuration hashes. Source attribution stays with the
public files. The current source catalogue registers only real data.

## Reviewed own recordings

Frame indices are zero-based and interval endpoints are inclusive.

| Video | Reviewed frames | Original match |
|---|---:|---|
| video01 | 0-199 | match01 |
| video02 | 0-188 | match02 |
| video03 | 0-699 | match03 |
| video04 | 0-699 | match04 |
| video05 | 0-159 | match05 |
| video06 | 0-159 | match06 |
| video07 | 0-172 | match07 |
| video08 | 0-151 | match08 |
| video09To16 | 0-179 | match09 |
| video09To16 | 180-359 | match10 |
| video09To16 | 360-529 | match11 |
| video09To16 | 530-709 | match07 |
| video09To16 | 710-889 | match12 |
| video09To16 | 890-1069 | match13 |
| video09To16 | 1070-1249 | match14 |

These are 3,684 reviewed frame records, 15 intervals, nine videos, and 14 original
matches. The original filename `jogo9-16.mp4` is preserved; it is not a declaration
that the montage contains eight new matches.

The original Jogo7 recording and montage frames 530-709 have the same
`matchId: match07`. They must stay together in future splits. Their possible
overlap in original footage has not been resolved, so their combined 353 records
must not be described as 353 verified unique images.

Each montage match consists of short cuts containing consecutive broadcast frames.
Exact cut boundaries and the mapping back to source broadcasts remain unresolved.
The catalog does not invent clip boundaries or timestamps. CVAT track identifiers,
especially imported single-frame tracks, are not certified player identities.

Only the listed intervals are eligible. Each own video declares
`annotationScope: completeForDeclaredClasses`, lists the 25 target names in
`classesAnnotated`, and records `annotationScopeEvidence: researcherDeclaration`.
This is the recorded review declaration for those intervals, not an automatic
verification of visual completeness. Class presence counts do not establish
visibility or prove visual absence.

Every interval has `internalCutsStatus: pending`. Its provenance distinguishes
catalogued reviewed limits from explicitly confirmed montage ranges. The interval
530-709 records `sameMatchAsVideoId: video07` and
`matchIdentityBasis: researcherConfirmedSameMatch`; this identity link is supplied
by the researcher rather than inferred from filenames or visual similarity.

## Target ontology and source labels

`configs/labelMap.yaml` defines 25 stable canonical class IDs:

- IDs 0-4: ball, two outfield-player team classes, and two goalkeeper team classes.
- IDs 5-24: twenty fixed court landmarks.

Athlete `TeamA` and `TeamB` identify teams within an original match, according
to the confirmed initial-side convention. Court `SideA` and `SideB` identify
fixed physical court sides. A team changing ends does not rename court landmarks.
Neither pair of names identifies a particular club across different matches.

CVAT mappings match `sourceLabel` after stripping outer whitespace and applying
Unicode case folding. This accommodates capitalization differences while keeping
the exact raw label in each annotation record. No numeric CVAT class ID is
invented from label declaration order. Raw box, ellipse, and point geometry is
preserved by the inventory; the canonical geometry is a target contract, not an
instruction to silently convert source shapes.

Mapping actions have distinct meanings:

- `map`: the source label has a canonical target class.
- `preserve`: retain the source semantic class without inventing a detailed target.
- `ignore`: exclude the label from target counts while retaining source provenance.

Referee labels use `ignore`. The generic `Jogador` and the additional
`Centro da Quadra` label use `preserve`; neither is silently assigned to a
different target. Unknown labels require an inventory finding rather than a guess.

## Public sources

The five required local sources use a flat layout: images and YOLO text labels
share a directory and pair by filename stem. Their numeric mappings are documented
in this project's `configs/labelMap.yaml`; the label text files themselves
do not declare class names.

| Source | Declared source classes | Local metadata |
|---|---|---|
| Futsal4400 | ball, player | None registered |
| SmartFutsal | ball, player | None registered |
| FutsalMOT | ball, goalkeeper, outfield_player, referee | README.dataset.txt; README.roboflow.txt |
| FutsalAnaliza | ball, goalkeeper, outfield_player, referee | README.dataset.txt; README.roboflow.txt |
| FutsalDetect | ball, goalkeeper, outfield_player, referee | README.dataset.txt; README.roboflow.txt |

The relevant `README*.txt` files are metadata, not annotation files. Archives
such as `Futsal4400/Dataset.zip` are not additional image sources. Source README
counts may describe a larger release than the local subset; record observations
without replacing them with published counts.

Ball maps to the canonical ball class. Generic player, goalkeeper, and outfield
player labels retain their source semantics with no invented team membership.
`annotationScope: sourceOntologyOnly` describes the available ontology; it
does not certify annotation completeness or declare missing target labels to be
negative examples. Empty labels must be reported as empty, not automatically
certified as true negatives.

Public sources are eligible only for compatible training stages. Their listed metadata is
evidence to inventory, not a grant of redistribution rights. Missing metadata
must remain explicit.

## Source integrity

Run `python -m futsal verifyRawSources --dataRoot data/raw` to check all imported
file hashes and exact membership. Raw files are excluded from Git and protected
from research output writes. The current source catalogue has no synthetic inputs.

## Split constraints and current coverage limit

The read-only inventory leaves split assignment unassigned. Curriculum preparation
now uses the frozen original-match assignment in `configs/splits.yaml`, with
eligible annotations for all 25 classes in training and test. The intended
proportions are approximately 70/15/15 by reviewed own frame records, with whole
original matches kept together. Public and synthetic sources do not enter that
denominator. See [dataset preparation](dataPreparation.md) for retained coverage
and quarantine policy.

Reviewed `lowerCornerSideB` annotations occur only in match09 (27 frames) and
match10 (62 frames). Future assignment must keep support in training and test;
validation support for this class is unavailable under the current grouping.
Putting both matches exclusively in test would remove all reviewed own examples
of that class from training. Splitting a match or its cuts would not create an
independent third match. Coverage is a property of annotations, not proof of
visual absence in other frames.

Known source findings require review before future materialization: some montage
frames contain two differently located annotations of the same landmark, and
some contain two ball boxes. This catalog preserves those inputs and does not
choose a replacement, delete a label, or treat duplication as resolved.
