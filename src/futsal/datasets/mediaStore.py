import os
import re
import uuid
from contextlib import contextmanager
from pathlib import Path

from futsal.datasets.datasetFiles import hashJson, readJson, writeJson
from futsal.inventory.catalog import hashFile, resolveSourcePath


def checkIdentity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Media identity must be a SHA256 digest.")
    return value


def verifyIdentity(value, field):
    if value[field] != hashJson({key: item for key, item in value.items() if key != field}):
        raise ValueError(f"Shared data {field} checksum mismatch.")


def atomicJson(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.partial")
    try:
        writeJson(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def storeLock(storeDir):
    root = Path(storeDir).resolve()
    project = Path(__file__).resolve().parents[3]
    protected = [project / name for name in ("src", "configs", ".git", "data/raw")]
    if Path(storeDir).is_symlink() or any(root.is_relative_to(path) or path.is_relative_to(root) for path in protected):
        raise ValueError("Media store must be separate from code, configuration and raw inputs.")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writeLock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError("Media store is being updated; retry after the writer finishes.") from error
    try:
        os.close(descriptor)
        yield root
    finally:
        lock.unlink()


def readBindings(storeDir):
    path = Path(storeDir) / "bindings.json"
    if not path.exists():
        return {"schemaVersion": 1, "assets": {}}
    bindings = readJson(path)
    if bindings.get("schemaVersion") != 1 or not isinstance(bindings.get("assets"), dict):
        raise ValueError("Unsupported media bindings.")
    return bindings


def resolveAsset(storeDir, assetId, *, bindings=None):
    checkIdentity(assetId)
    bindings = readBindings(storeDir) if bindings is None else bindings
    item = bindings["assets"].get(assetId)
    if item is None:
        raise ValueError(f"Missing media asset binding: {assetId}.")
    return resolveSourcePath(Path(item["root"]), item["path"])


def verifyAssets(storeDir, assets):
    bindings = readBindings(storeDir)["assets"]
    for assetId, expected in assets.items():
        checkIdentity(assetId)
        item = bindings.get(assetId)
        if item is None:
            raise ValueError(f"Missing media asset binding: {assetId}.")
        path = resolveSourcePath(Path(item["root"]), item["path"])
        if path.stat().st_size != expected["byteSize"] or hashFile(path) != assetId:
            raise ValueError(f"Shared media asset changed: {assetId} ({path}).")
    return {"assetCount": len(assets), "hashesVerified": True}


def bindAsset(bindings, assetId, path):
    checkIdentity(assetId)
    path = Path(path).resolve()
    if hashFile(path) != assetId:
        raise ValueError(f"Image differs from its declared content hash: {path}.")
    previous = bindings["assets"].get(assetId)
    if previous is not None:
        original = resolveSourcePath(Path(previous["root"]), previous["path"])
        if not original.is_file() or hashFile(original) != assetId:
            raise ValueError(f"Existing media binding is invalid; explicitly rebind {assetId}.")
        return
    bindings["assets"][assetId] = {"root": str(path.parent), "path": path.name}


def readCollection(storeDir, collectionId):
    checkIdentity(collectionId)
    value = readJson(Path(storeDir) / "collections" / f"{collectionId}.json")
    verifyIdentity(value, "collectionId")
    if value.get("kind") != "sharedMediaCollection" or value.get("schemaVersion") != 1:
        raise ValueError("Unsupported media collection.")
    return value


def publishCollection(root, collection, bindings):
    collection["collectionId"] = hashJson(collection)
    directory = root / "collections"
    directory.mkdir(exist_ok=True)
    path = directory / f"{collection['collectionId']}.json"
    if path.exists() and readJson(path) != collection:
        raise ValueError("Immutable media collection already exists with different content.")
    atomicJson(root / "bindings.json", bindings)
    if not path.exists():
        atomicJson(path, collection)
    return collection


def storeCollections(storeDir):
    return [readCollection(storeDir, path.stem)
            for path in sorted((Path(storeDir) / "collections").glob("*.json"))]


def rebindMedia(storeDir, assetId, root, relativePath):
    path = resolveSourcePath(Path(root).resolve(), relativePath)
    checkIdentity(assetId)
    if hashFile(path) != assetId:
        raise ValueError("Replacement media binding has different bytes.")
    with storeLock(storeDir) as store:
        bindings = readBindings(store)
        if assetId not in bindings["assets"]:
            raise ValueError("Cannot rebind an unregistered asset.")
        bindings["assets"][assetId] = {"root": str(Path(root).resolve()), "path": relativePath}
        bindings["providers"] = sorted(set(bindings.get("providers", [])) | {str(Path(root).resolve())})
        atomicJson(store / "bindings.json", bindings)


def protectMediaOutput(outputDir, storeDir):
    from futsal.datasets.datasetFiles import validateNewOutput

    for provider in readBindings(storeDir).get("providers", []):
        validateNewOutput(outputDir, provider, [provider])


def isSharedView(datasetDir):
    return readJson(Path(datasetDir) / "contract.json").get("kind") == "sharedTaskView"
