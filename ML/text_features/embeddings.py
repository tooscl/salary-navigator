"""multilingual-e5 sentence embeddings with an on-disk cache."""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np


def e5_embeddings(texts: list[str], model_name: str, cache_dir: Path, log, batch_size: int = 32,
                  max_length: int = 512) -> np.ndarray:
    """L2-normalised mean-pooled embeddings of "passage: <text>"; cached by content hash."""
    digest = hashlib.sha1(f"{model_name}|{max_length}".encode("utf-8"))
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{model_name.replace('/', '__')}_{digest.hexdigest()[:16]}.npy"
    if path.exists():
        log(f"e5: cached embeddings {path.name}")
        return np.load(path)

    import torch
    from transformers import AutoModel, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        log(f"e5: {model_name} loaded from the local Hugging Face cache")
    except Exception as error:  # not cached yet
        log(f"e5: not in local cache ({type(error).__name__}); downloading {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    log(f"e5: encoding {len(texts)} texts on {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    order = np.argsort([len(t) for t in texts])
    result = None
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            index = order[start:start + batch_size]
            batch = tokenizer(["passage: " + texts[i] for i in index], padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy()
            if result is None:
                result = np.zeros((len(texts), pooled.shape[1]), dtype=np.float32)
            result[index] = pooled
            if (start // batch_size) % 25 == 0:
                log(f"e5: {min(start + batch_size, len(texts))}/{len(texts)}")
    np.save(path, result)
    return result
