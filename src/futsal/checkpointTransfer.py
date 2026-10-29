import hashlib


def tensorHash(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def classificationKeys(model):
    head = model.model[-1]
    prefix = next(name for name, module in model.named_modules() if module is head)
    keys = []
    for branch in ("cv3", "one2one_cv3"):
        layers = getattr(head, branch, None)
        if layers is None:
            raise ValueError(f"YOLO26 training requires the {branch} classification branch.")
        for index, sequence in enumerate(layers):
            if sequence[-1].out_channels != len(model.names):
                raise ValueError("Classification head size differs from target class names.")
            keys.extend(f"{prefix}.{branch}.{index}.{len(sequence) - 1}.{parameter}"
                        for parameter in ("weight", "bias"))
    return keys


def loadAudited(model, source):
    import torch

    sourceNames, targetNames = dict(source.names), dict(model.names)
    sourceState = {key: value.detach().float().cpu().clone()
                   for key, value in source.state_dict().items()}
    sourceKeys = classificationKeys(source)
    model.load(source, verbose=False)
    if model.names != targetNames:
        raise ValueError("Native weight loading changed the target class ontology.")
    targetState = model.state_dict()
    targetKeys = classificationKeys(model)
    if set(targetKeys) != set(sourceKeys):
        raise ValueError("Source and destination classification branches are incompatible.")
    sourceLookup = {name.strip().lower(): index for index, name in sourceNames.items()}
    matched = {index: sourceLookup[name.strip().lower()] for index, name in targetNames.items()
               if name.strip().lower() in sourceLookup}
    rows = []
    for key in targetKeys:
        for targetIndex, sourceIndex in matched.items():
            before, after = sourceState[key][sourceIndex], targetState[key][targetIndex].float().cpu()
            if not torch.equal(before, after):
                raise ValueError(f"Class-name transfer failed for {targetNames[targetIndex]}: {key}.")
            rows.append({"tensor": key, "className": targetNames[targetIndex],
                         "sourceIndex": sourceIndex, "targetIndex": targetIndex,
                         "sourceSha256": tensorHash(before), "targetSha256": tensorHash(after)})
        if key.endswith("weight"):
            for index in set(targetNames) - set(matched):
                if index < len(sourceNames) and sourceState[key][index].shape == targetState[key][index].shape:
                    if torch.equal(sourceState[key][index], targetState[key][index].float().cpu()):
                        raise ValueError(f"Unmatched class was copied by index: {targetNames[index]}.")
    equalKeys = [key for key, value in targetState.items() if key not in targetKeys
                 and key in sourceState and value.shape == sourceState[key].shape
                 and torch.equal(value.float().cpu(), sourceState[key])]
    compatibleKeys = [key for key, value in targetState.items() if key not in targetKeys
                      and key in sourceState and value.shape == sourceState[key].shape]
    if not equalKeys or set(equalKeys) != set(compatibleKeys):
        raise ValueError("Compatible backbone or hidden head tensors were not preserved.")
    return {"sourceNames": sourceNames, "targetNames": targetNames,
            "verifiedCompatibleTensorCount": len(equalKeys), "compatibleTensorKeys": equalKeys,
            "targetTensorCount": len(targetState), "classificationTensorKeys": targetKeys,
            "matchedClasses": [targetNames[index] for index in matched],
            "newClasses": [name for index, name in targetNames.items() if index not in matched],
            "verifiedRows": rows, "verifiedBeforeFirstBatch": True}
