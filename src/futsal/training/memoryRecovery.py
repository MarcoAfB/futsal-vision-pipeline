def errorCauses(error):
    causes, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        causes.append({"type": type(error).__name__, "module": type(error).__module__, "message": str(error)})
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return causes


def isMemoryFailure(causes):
    for cause in causes:
        text = cause["message"].lower()
        if cause["type"] == "MemoryError" or (
            cause["type"] in {"OutOfMemoryError", "AcceleratorError"} and
            cause["module"].startswith("torch") and "out of memory" in text
        ):
            return True
        if cause["type"] in {"RuntimeError", "AcceleratorError"} and any(token in text for token in (
            "cuda out of memory", "cuda error: out of memory", "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed", "defaultcpuallocator: can't allocate memory",
        )):
            return True
    return False


def requireFixedBatch(batch, requested, retries):
    if batch != requested or retries:
        raise RuntimeError("Fixed batch recovery refused a native retry; restart this stage in a new audited attempt.")
