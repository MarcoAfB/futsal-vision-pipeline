# Initial dataset audit

Run from the project root:

```text
uv run python -m futsal inventory --dataRoot ../Python --outputDir outputs/initialInventory
```

The audit completed with exit code 1: reports were generated and source findings
remain for review. Full JSON and Markdown results are local generated artifacts;
this document records the observations and their interpretation.

## Reviewed own data

The catalog and authoritative reviewed limits agree: **3,684 frame records,
14 original matches, nine readable videos, and 15 inclusive reviewed intervals**.
Video length and extra XML observations do not expand these intervals.

The old Jogo7 prefix and montage frames 530-709 share `match07`, totaling 353
frame records. Possible overlap in the original footage remains unresolved.
Records must not be described as verified unique images.

All 25 target classes occur in the reviewed own annotations. `lowerCornerSideB`
occurs in 89 records across only `match09` (27) and `match10` (62). This prevents
independent match support in all three splits. Future assignment must preserve
training and test support; validation results for that class will be unsupported.
No split was assigned, and coverage includes observations with unresolved findings.

## External sources

| Source | Images | Empty label files | Availability |
|---|---:|---:|---|
| Futsal4400 | 2,768 | 3 | Partial: one degenerate box |
| SmartFutsal | 637 | 0 | Available |
| FutsalMOT | 404 | 0 | Available |
| FutsalAnaliza | 904 | 2 | Available; metadata count differs |
| FutsalDetect | 706 | 0 | Partial: five out-of-bounds boxes |
| Synthetic court | 20 | 0 | Available |

There are 5,419 public images and 20 synthetic images. These are separate from
the own-data frame denominator and are training-only. Availability describes
successful file inspection, not annotation correctness or model readiness.

The synthetic source contains 138 five-field YOLO box annotations representing
court landmarks. Its initial three-field declaration caused 138 format findings;
the corrected registration matches the actual files without converting them.

## Findings requiring review

| Finding | Count | Location or interpretation |
|---|---:|---|
| Own geometry beyond image bounds | 22 | Ball ellipses in the reviewed Jogo4 interval |
| Public box edges beyond normalized bounds | 5 | FutsalDetect; file paths and lines in JSON |
| Degenerate public box | 1 | Futsal4400/g10535.txt, line 9 |
| Repeated landmark in one frame | 13 | 12 montage frames and Jogo4 frame 243 |
| Multiple ball annotations in one frame | 22 | Montage; observations preserved |
| Sources containing empty label files | 2 | Futsal4400 and FutsalAnaliza; five files total |
| Metadata image-count disagreement | 1 | FutsalAnaliza declares 973; 904 locally present |
| Limited match coverage | 1 | lowerCornerSideB |
| Exact cut boundaries pending | 1 | No continuity inferred between cuts |
| Frame uniqueness unverified | 1 | No decoded-frame deduplication performed |

The result contains **28 error findings and 41 warnings**. These refer to source
observations and dataset limitations, rather than a failed inventory execution.
Partially visible objects and multiple annotations need contextual review before
choosing any future clipping, selection, or conversion policy. No label was corrected,
discarded, or certified as a true negative by this audit.

## Provenance and verification

The inventory records SHA-256 hashes of annotations, catalogs, the reviewed-limit
registry, and registered source evidence. Source paths are relative to `dataRoot`.
Configuration hashes refer to exact bytes, including line endings.

Before/after snapshots matched for 5,459 annotation/evidence files and the file size
and modification time of 5,448 media files. Media content was not fully hashed;
these checks do not establish perceptual uniqueness or tracking identity.

The verification suite uses synthetic fixtures and passes without the real datasets.
It covers parsing, reviewed-range boundaries, source mappings, match grouping,
geometry findings, deterministic reports, source preservation, and CLI output guards.
Training, model selection, temporal analyses, and sports statistics remain subsequent
stages. Annotation ambiguities and exact cut boundaries require review before the
corresponding future dataset materialization or temporal analysis.
