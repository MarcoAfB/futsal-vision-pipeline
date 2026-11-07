# Independent raw sources

The project's input root is data/raw. It contains only the real inputs declared
in configs/dataSources.yaml, using its reviewed intervals and match identities.

```text
data/raw/
  own/videos/           nine registered recordings
  own/annotations/      their nine current CVAT XMLs
  public/sourceId/      original public images, labels and source READMEs
  sourceManifest.json   exact file membership, sizes and SHA-256
```

The source catalogue uses schemaVersion2. Reviewed limits are declared directly
in this project, and annotation mapping evidence comes from configs/labelMap.yaml.
No external code or review/configuration registry is required. Public record
namespaces retain stable sample IDs when physical source directories change.
The imported snapshot currently contains10862 files and1327130454 raw content
bytes, excluding its manifest. It includes all3684 reviewed own records across
14 original matches,9 videos and15 intervals; public files remain training-only.
Original media/annotation bytes remain unchanged, including quarantined inputs.

Verify and inspect from the project root:

```text
python -m futsal verifyRawSources --dataRoot data/raw
python -m futsal inventory --dataRoot data/raw --outputDir outputs/sourceInventory
python -m futsal planDataset --dataRoot data/raw --outputDir outputs/nativeSourcePlan
```

Output destinations must be new and outside the raw root. Source integrity checks
refuse missing/extra files, changed bytes, broken bindings and links. No trained
weights, experimental results, synthetic data or old task exports are raw inputs.
Raw files and their manifest are ignored by Git and must be transferred separately
from code when a destination needs source reconstruction. Prepared training bundles
remain self-contained and do not require raw sources on the training server.

Existing project dataset/projection contracts are preserved. Native path bindings
allow a verified source-schema transition for projections from an existing frozen
bundle only when reviewed membership, ontology, mappings, namespaces and registered
raw bytes match. This does not authorize different sources or changed review limits.
Do not edit a sealed plan to adopt current paths; new plans get their own identity.

`importRawSources` copies files according to an explicit source catalogue and root,
verifies both sides and publishes a new raw tree. Its --catalogOutput writes the
corresponding native catalogue to a separate new file for review. It never replaces
the active configuration or removes the originals automatically. Existing targets
are refused; retry a failed import using a new destination.

The source videos/XMLs also preserve the inputs for the reserved test. This import
does not export test images, expand reviewed ranges or execute any model evaluation.
