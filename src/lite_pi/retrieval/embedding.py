"""Pinned local E5 representation. Optional native runtimes load only on explicit use."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

from lite_pi.knowledge.db import KnowledgeError
from lite_pi.runtime import PATHS
from lite_pi.tools.contracts import JsonObject

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DEFAULT_MODEL_DIR = PATHS.state / "models/multilingual-e5-small" / REVISION
Mode = Literal["bm25", "semantic", "hybrid"]


@dataclass(frozen=True)
class Representation:
    model: str = MODEL
    revision: str = REVISION
    dimension: int = 384
    pooling: str = "masked_mean"
    normalization: str = "l2"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    max_tokens: int = 512
    chunking: str = "locatable-lines-800-v1-no-truncation"

    def payload(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class Encoding:
    vectors: tuple[tuple[float, ...], ...]
    wall_seconds: float
    cpu_seconds: float
    token_lengths: tuple[int, ...]

    def metrics(self) -> JsonObject:
        return {
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "token_lengths": list(self.token_lengths),
            "vectors": len(self.vectors),
        }


class Encoder(Protocol):
    representation: Representation

    def encode(self, texts: Sequence[str], *, query: bool, stopped: Callable[[], bool]) -> Encoding: ...
    def close(self) -> None: ...


def validate_vector(vector: object, dimension: int) -> tuple[float, ...]:
    if (
        not isinstance(vector, (list, tuple))
        or len(vector) != dimension
        or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vector
        )
    ):
        raise KnowledgeError("embedding_dimension_or_nonfinite")
    norm = math.sqrt(sum(v * v for v in vector))
    if abs(norm - 1) > 0.0001:
        raise KnowledgeError("embedding_not_normalized")
    return tuple(vector)


class LocalEncoder:
    representation = Representation()

    def __init__(self, directory: Path) -> None:
        import json

        required = ("model.onnx", "tokenizer.json", "representation.json")
        if not all((directory / name).is_file() for name in required):
            raise KnowledgeError(f"embedding_model_missing:{directory}")
        metadata = json.loads((directory / "representation.json").read_text())
        if (
            metadata.get("model") != MODEL
            or metadata.get("revision") != REVISION
            or metadata.get("dimension") != 384
        ):
            raise KnowledgeError("embedding_model_revision_mismatch")
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as error:
            raise KnowledgeError("embedding_dependencies_missing:uv sync --extra embeddings") from error
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self._session: ort.InferenceSession | None = ort.InferenceSession(
            str(directory / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self._tokenizer.no_truncation()
        self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

    def encode(self, texts: Sequence[str], *, query: bool, stopped: Callable[[], bool]) -> Encoding:
        import numpy as np

        if self._session is None:
            raise KnowledgeError("embedding_closed")
        wall, cpu = time.monotonic(), time.process_time()
        vectors: list[tuple[float, ...]] = []
        lengths: list[int] = []
        prefix = self.representation.query_prefix if query else self.representation.passage_prefix
        # Bounded batches keep CPU memory predictable; cancellation is checked between native calls.
        for start in range(0, len(texts), 4):
            if stopped():
                raise KnowledgeError("cancelled")
            encoded = self._tokenizer.encode_batch([prefix + t for t in texts[start : start + 4]])
            actual_lengths = [sum(e.attention_mask) for e in encoded]
            if any(length > self.representation.max_tokens for length in actual_lengths):
                raise KnowledgeError("embedding_input_too_long:no_truncation")
            inputs = {
                "input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask for e in encoded], dtype=np.int64),
                "token_type_ids": np.asarray([e.type_ids for e in encoded], dtype=np.int64),
            }
            hidden = np.asarray(self._session.run(["last_hidden_state"], inputs)[0], dtype=np.float32)
            mask = inputs["attention_mask"][:, :, None]
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
            vectors.extend(validate_vector(row.tolist(), self.representation.dimension) for row in pooled)
            lengths.extend(actual_lengths)
            if stopped():
                raise KnowledgeError("cancelled")
        return Encoding(tuple(vectors), time.monotonic() - wall, time.process_time() - cpu, tuple(lengths))

    def close(self) -> None:
        self._session = None
