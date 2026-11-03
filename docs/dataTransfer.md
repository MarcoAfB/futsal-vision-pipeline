# Dataset transport

The prepared bundle is immutable. Transfer its contracted files as one archive,
without resizing/recompressing images or rewriting labels/YAMLs. The scripts use
only the Python 3.12/3.13 standard library. Keep the three transport scripts
together in a Git checkout; no model environment is required for extraction.

First run the complete semantic verifier in the project environment and save an
external receipt. Replace the example contract ID with the independently retained
ID of the bundle being sent:

```text
uv run --locked python -m futsal verifyDataset --datasetDir data/prepared/curriculumV1 --receipt outputs/verification/datasetVerification.json
python scripts/datasetTransfer.py pack --datasetDir data/prepared/curriculumV1 --contractId <contractId> --verification outputs/verification/datasetVerification.json --archive data/transfers/curriculumV1.tar.gz --receipt data/transfers/curriculumV1.receipt.json
```

The packer requires a successful full-verification receipt bound to this contract,
exact membership, ordinary files and real registered development sources. It
checks hashes while streaming, checks the complete archive before publication,
and writes archive SHA-256, file hashes/counts/sizes, dataset/plan IDs, source policy
and tool hashes to the receipt. `--compression none` writes an uncompressed tar;
otherwise gzip level 1 is used. Most input bytes are already compressed images;
the benefit is transferring one file, without promising a large size reduction.

Send archive and receipt using the existing SFTP/rsync workflow. Retain sender
hashes independently: an internal checksum is not a signature. On receipt:

```text
python3.12 scripts/datasetTransfer.py check --archive incoming/curriculumV1.tar.gz --receipt incoming/curriculumV1.receipt.json
python3.12 scripts/datasetTransfer.py unpack --archive incoming/curriculumV1.tar.gz --receipt incoming/curriculumV1.receipt.json --outputDir data/prepared/curriculumV1
uv run --locked python -m futsal verifyDataset --datasetDir data/prepared/curriculumV1 --receipt outputs/verification/receivedDataset.json
```

Extraction verifies the complete archive hash before reading members, rejects
unsafe paths, case collisions, duplicate/missing/extra members, links, devices and
size/hash mismatches. It writes into a short ordinary staging directory beside
the target, checks membership and identities, then publishes a new target.
Existing destinations are never overwritten. Ownership and ACLs are inherited
locally; archive owner metadata is not applied. Failure leaves a named partial
and external failure record, never a ready target/receipt.

Both packing and extraction reserve 2 GiB beyond their estimated output bytes.
Also budget environment/cache space, checkpoints, reports and one complete runtime
copy per retained run/continuation. Do not attempt a full local extraction when
only enough disk remains for the archive. No automatic cleanup removes old data.

The package preserves contracted test eligibility/native annotation metadata but
contains no exported test images, YOLO test labels or test galleries. It does not
include raw legacy data, source videos, code, weights or experiments. Transfer
code through Git (or an offline Git bundle) and the official initial weights with
their provenance sidecar independently. Generated archives, receipts, scientific
data and machine settings belong under ignored data/outputs or local settings.

## Returning results

Package each finished attempt separately. Repeat `--reportDir` for additional
sealed evaluations/presentations, and optionally include its launcher logs:

```text
python3.12 scripts/resultTransfer.py pack --runDir runs/seed42 --reportDir reports/seed42/<evaluation> --launchDir launches/seed42 --archive outgoing/seed42.tar.gz --receipt outgoing/seed42.receipt.json
python scripts/resultTransfer.py check --archive incoming/seed42.tar.gz --receipt incoming/seed42.receipt.json
python scripts/resultTransfer.py unpack --archive incoming/seed42.tar.gz --receipt incoming/seed42.receipt.json --outputDir outputs/imported/seed42
```

The new import contains `run/`, numbered `reports/report01/` folders and optional
`launch/`. Its receipt maps report folder names to the original names. Included
run inputs/code, best/last checkpoints, stage audits, losses, metrics and logs
are hashed. Evaluation seals and presentation artifact hashes are checked before
packing and after extraction; a report bound to different run metadata is refused.
Training metadata and server absolute paths are preserved byte for byte.

`runtimeData`, backend caches and evaluationWork are excluded explicitly.
`runtimeEvidence.json` records their original provenance; import does not claim
verification of omitted bytes. The original server attempt stays intact. A sealed
report can be regenerated with `reportRun --evaluationDir <import>/reports/report01
--outputDir <newPresentation>` without inference. A new evaluation reconstructs
runtime copies from the matching prepared bundle and saved selection.
