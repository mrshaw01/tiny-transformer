from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PackedDatasetMeta:
    seq_len: int
    dtype: str
    vocab_size: int
    tokenizer_name: str
    eos_token_id: int
    bos_token_id: Optional[int] = None

    @staticmethod
    def load(path: str | Path) -> "PackedDatasetMeta":
        raw = json.loads(Path(path).read_text())
        return PackedDatasetMeta(
            seq_len=int(raw["seq_len"]),
            dtype=str(raw["dtype"]),
            vocab_size=int(raw["vocab_size"]),
            tokenizer_name=str(raw["tokenizer_name"]),
            eos_token_id=int(raw["eos_token_id"]),
            bos_token_id=int(raw["bos_token_id"]) if raw.get("bos_token_id") is not None else None,
        )


class PackedMemmapDataset(Dataset):

    def __init__(self, bin_path: str | Path, meta: PackedDatasetMeta):
        self.bin_path = Path(bin_path)
        self.meta = meta
        self._data = np.memmap(self.bin_path, dtype=np.dtype(self.meta.dtype), mode="r")
        if self._data.size % self.meta.seq_len != 0:
            raise ValueError(
                f"{self.bin_path} token count ({self._data.size}) not divisible by seq_len ({self.meta.seq_len})")
        self._num_sequences = self._data.size // self.meta.seq_len

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.meta.seq_len
        end = start + self.meta.seq_len
        input_ids = torch.from_numpy(self._data[start:end].astype(np.int64, copy=False))
        return {"input_ids": input_ids, "labels": input_ids}


def default_data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([f["input_ids"] for f in features], dim=0)
    labels = torch.stack([f["labels"] for f in features], dim=0)
    return {"input_ids": input_ids, "labels": labels}
