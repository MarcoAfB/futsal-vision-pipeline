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
