import hashlib
from collections import Counter, defaultdict


def sourceComposition(rows):
    groups = defaultdict(Counter)
    for row in rows:
        counts = groups[(row["sourceId"], row["origin"])]
        counts["imageCount"] += 1
        counts["positiveImages"] += bool(row["targets"])
        counts["negativeImages"] += not row["targets"]
        counts["targetCount"] += len(row["targets"])
    return [{"sourceId": source, "origin": origin, **dict(counts)}
            for (source, origin), counts in sorted(groups.items())]


def verifyNaturalSampling(loader, imageCount):
    from torch.utils.data import RandomSampler

    sampler = loader.sampler
    if not isinstance(sampler, RandomSampler) or sampler.replacement or (
        sampler.num_samples != imageCount or loader.drop_last
    ):
        raise ValueError("Natural mixed training requires shuffled sampling without replacement or dropped batches.")
    return {"policy": "naturalMixedSources", "shuffle": True, "replacement": False,
            "dropLast": False, "imagesPerEpoch": imageCount}


class EpochExposure:
    def __init__(self, rows):
        self.rows = {row["recordId"]: row for row in rows}
        self.expected = Counter(self.rows.keys())
        self.completedEpochs = 0

    def begin(self, epoch):
        self.epoch = epoch
        self.observed = Counter()
        self.order = hashlib.sha256()

    def observe(self, recordIds):
        if any(record not in self.rows for record in recordIds):
            raise ValueError("Training emitted a record outside the immutable selection.")
        self.observed.update(recordIds)
        for record in recordIds:
            self.order.update((record + "\n").encode())

    def finish(self):
        if self.observed != self.expected:
            raise ValueError("Each natural mixed epoch must consume every selected training record exactly once.")
        self.completedEpochs += 1
        return {"epoch": self.epoch, "imageCount": sum(self.observed.values()),
                "uniqueRecordCount": len(self.observed), "orderSha256": self.order.hexdigest(),
                "bySource": sourceComposition(self.rows.values()), "coverageVerified": True}
