import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

toolVersion = 1
reserveBytes = 2 * 1024 ** 3


def readJson(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def writeJson(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        output.write("\n")


def hashJson(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def hashFile(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def safeRelative(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or any(ord(c) < 32 for c in name):
        raise ValueError(f"Unsafe archive path: {name!r}.")
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or path.as_posix() != name or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe archive path: {name!r}.")
    for part in path.parts:
        if part.endswith((".", " ")) or re.fullmatch(r"(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?", part, re.I):
            raise ValueError(f"Nonportable archive path: {name!r}.")
    return path


def validateFiles(files):
    expected, prefixes = {}, {}
    for item in files:
        path = safeRelative(item["path"])
        if type(item["byteSize"]) is not int or item["byteSize"] < 0 or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
            raise ValueError("Invalid file size or SHA-256 in receipt.")
        if item["path"] in expected:
            raise ValueError("Duplicate file in receipt.")
        for depth in range(1, len(path.parts) + 1):
            prefix = "/".join(path.parts[:depth])
            key = prefix.casefold()
            if key in prefixes and prefixes[key] != prefix:
                raise ValueError("Case-colliding archive paths are not portable.")
            prefixes[key] = prefix
        expected[item["path"]] = item
    if not expected:
        raise ValueError("An archive must contain files.")
    for path in expected:
        if any(parent.as_posix() in expected for parent in PurePosixPath(path).parents if parent.as_posix() != "."):
            raise ValueError("Archive file/directory paths collide.")
    return expected


def regularFiles(root):
    root = Path(root)
    if root.is_symlink() or root.is_junction():
        raise ValueError("Input roots must not be links.")
    result = {}
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if path.is_symlink() or path.is_junction() or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"Only ordinary directories and regular files are permitted: {path}.")
        if stat.S_ISREG(mode):
            result[path.relative_to(root).as_posix()] = path
    return result


def requireSpace(directory, size):
    ancestor = Path(directory).resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    if free < size + reserveBytes:
        raise ValueError(f"Insufficient disk space: need {size} bytes plus {reserveBytes} reserve; have {free}.")


def requireNewPath(path, protected=()):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink() or path.is_junction():
        raise ValueError(f"Destination must be new: {path}.")
    path = path.resolve()
    for source in protected:
        source = Path(source).resolve()
        if path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("Destination must be outside and must not contain inputs.")
    return path


def verifyReceipt(receipt, kind=None):
    if receipt.get("schemaVersion") != 1 or receipt.get("toolVersion") != toolVersion:
        raise ValueError("Unsupported transfer receipt.")
    if receipt.get("receiptId") != hashJson({k: v for k, v in receipt.items() if k != "receiptId"}):
        raise ValueError("Transfer receipt checksum mismatch.")
    if kind is not None and receipt["kind"] != kind:
        raise ValueError("Unexpected transfer package kind.")
    expected = validateFiles(receipt["files"])
    if receipt["fileCount"] != len(expected) or receipt["uncompressedBytes"] != sum(item["byteSize"] for item in expected.values()):
        raise ValueError("Receipt count or size mismatch.")
    return expected


class HashingReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()

    def read(self, size):
        value = self.source.read(size)
        self.digest.update(value)
        return value


def checkArchive(archive, receipt, outputDir=None, kind=None):
    archive = Path(archive)
    expected = verifyReceipt(receipt, kind)
    if archive.stat().st_size != receipt["archiveBytes"] or hashFile(archive) != receipt["archiveSha256"]:
        raise ValueError("Archive size or SHA-256 mismatch.")
    seen = set()
    with tarfile.open(archive, mode="r|*") as source:
        for member in source:
            safeRelative(member.name)
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError("Unexpected, duplicate or nonregular archive member.")
            item = expected[member.name]
            if member.size != item["byteSize"]:
                raise ValueError("Archive member size differs from receipt.")
            seen.add(member.name)
            digest = hashlib.sha256()
            target = None
            if outputDir is not None:
                path = Path(outputDir) / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                target = path.open("xb")
            try:
                with source.extractfile(member) as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                        if target is not None:
                            target.write(chunk)
            finally:
                if target is not None:
                    target.close()
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Archive member hash mismatch: {member.name}.")
    if seen != set(expected):
        raise ValueError("Archive has missing files.")
    return {"fileCount": len(seen), "hashesVerified": True, "receiptId": receipt["receiptId"]}


def publishFile(partial, final):
    os.link(partial, final)
    partial.unlink()


def packArchive(sources, archivePath, receiptPath, kind, metadata, compression="gzip", protected=()):
    if compression not in {"gzip", "none"}:
        raise ValueError("Compression must be gzip or none.")
    archive = requireNewPath(archivePath, protected)
    receiptPath = requireNewPath(receiptPath, [*protected, archive])
    partial = requireNewPath(archive.with_name(archive.name + ".partial"), [*protected, receiptPath])
    files = [{"path": name, "byteSize": item["byteSize"], "sha256": item["sha256"]}
             for name, (_, item) in sorted(sources.items())]
    validateFiles(files)
    size = sum(item["byteSize"] for item in files)
    requireSpace(archive.parent, size + len(files) * 2048 + size // 100)
    archive.parent.mkdir(parents=True, exist_ok=True)
    receiptPath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with partial.open("xb") as destination:
            options = {"compresslevel": 1} if compression == "gzip" else {}
            with tarfile.open(fileobj=destination, mode="w|gz" if compression == "gzip" else "w|", **options) as output:
                for item in files:
                    path, _ = sources[item["path"]]
                    if not stat.S_ISREG(path.lstat().st_mode) or path.is_junction() or path.stat().st_size != item["byteSize"]:
                        raise ValueError("Input type or size changed before packing.")
                    info = tarfile.TarInfo(item["path"])
                    info.size, info.mode = item["byteSize"], 0o644
                    with path.open("rb") as source:
                        reader = HashingReader(source)
                        output.addfile(info, reader)
                    if reader.digest.hexdigest() != item["sha256"]:
                        raise ValueError(f"Input changed while packing: {item['path']}.")
        receipt = {"schemaVersion": 1, "toolVersion": toolVersion, "kind": kind,
                   "createdAt": datetime.now(UTC).isoformat(), "compression": compression,
                   "archiveBytes": partial.stat().st_size, "archiveSha256": hashFile(partial),
                   "uncompressedBytes": size, "fileCount": len(files), "files": files, "metadata": metadata}
        receipt["receiptId"] = hashJson(receipt)
        checkArchive(partial, receipt)
        receiptPartial = requireNewPath(receiptPath.with_name(receiptPath.name + ".partial"), [*protected, archive, partial])
        writeJson(receiptPartial, receipt)
        publishFile(partial, archive)
        publishFile(receiptPartial, receiptPath)
        return receipt
    except (Exception, KeyboardInterrupt) as error:
        writeJson(archive.with_name(archive.name + f".failure-{uuid.uuid4().hex}.json"),
                  {"status": "failed", "error": str(error), "partialArchive": str(partial)})
        raise


def unpackArchive(archivePath, receiptPath, outputDir, kind, validate):
    archive, receiptPath = Path(archivePath).resolve(), Path(receiptPath).resolve()
    receipt = readJson(receiptPath)
    verifyReceipt(receipt, kind)
    output = requireNewPath(outputDir, [archive, receiptPath])
    requireSpace(output.parent, receipt["uncompressedBytes"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".partial-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        checkArchive(archive, receipt, staging, kind)
        if set(regularFiles(staging)) != {item["path"] for item in receipt["files"]}:
            raise ValueError("Extracted membership differs from receipt.")
        validate(staging, receipt)
        requireNewPath(output, [archive, receiptPath])
        staging.rename(output)
        return receipt
    except (Exception, KeyboardInterrupt) as error:
        writeJson(staging.with_name(staging.name + ".failure.json"),
                  {"status": "failed", "error": str(error), "partialDirectory": str(staging)})
        raise
