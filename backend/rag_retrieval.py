from collections.abc import Callable
from dataclasses import dataclass
import math
import re
from typing import Any
from uuid import UUID

from backend.embeddings import EmbeddingError, generate_query_embedding
from backend.rag_preparation import MetadataValue
from backend.vector_store import ChromaVectorStore, VectorStoreUnavailableError

DEFAULT_TOP_K = 5
MAX_TOP_K = 50
VECTOR_CANDIDATE_COUNT = 30
LEXICAL_CANDIDATE_COUNT = 30
LEXICAL_WEIGHT = 0.7
VECTOR_WEIGHT = 0.3
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "catalog",
        "could",
        "find",
        "for",
        "include",
        "includes",
        "mention",
        "mentions",
        "name",
        "names",
        "of",
        "product",
        "products",
        "related",
        "the",
        "to",
        "what",
        "which",
        "whose",
        "with",
    }
)


class RetrievalError(RuntimeError):
    """Base error for semantic retrieval."""


class InvalidRetrievalRequestError(RetrievalError):
    """Raised when query parameters are invalid."""


class DatasetNotIndexedError(RetrievalError):
    """Raised when an upload has no chunks in the vector store."""


class RetrievalUnavailableError(RetrievalError):
    """Raised when embeddings or vector search are unavailable."""


@dataclass(frozen=True)
class RetrievalMatch:
    rank: int
    chunk_id: str
    document: str
    source_row: int | None
    metadata: dict[str, MetadataValue]
    similarity_score: float


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, MetadataValue]:
    return {
        str(key): value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
        and not str(key).lower().endswith(("_path", "filepath"))
        and str(key).lower() != "path"
    }


def _first_query_results(result: dict[str, Any], key: str) -> list[Any]:
    values = result.get(key)
    if not isinstance(values, list) or not values or not isinstance(values[0], list):
        raise RetrievalUnavailableError("Vector search returned an invalid result.")
    return values[0]


def _stem_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _query_terms(query: str) -> list[str]:
    terms = []
    for token in _TOKEN_PATTERN.findall(query.casefold()):
        if token in _QUERY_STOP_WORDS:
            continue
        stemmed = _stem_token(token)
        if stemmed not in terms:
            terms.append(stemmed)
    return terms


def _document_field(document: str, label: str) -> str:
    match = re.search(
        rf"^-\s*{re.escape(label)}:\s*(.+)$",
        document,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _lexical_score(query_terms: list[str], metadata: dict[str, Any], document: str) -> float:
    if not query_terms:
        return 0.0
    fields = (
        (str(metadata.get("product_name") or _document_field(document, "Product Name")), 1.0),
        (str(metadata.get("sub_category") or _document_field(document, "Sub-Category")), 0.65),
        (str(metadata.get("category") or _document_field(document, "Category")), 0.45),
    )
    score = 0.0
    for term in query_terms:
        best = 0.0
        for value, weight in fields:
            tokens = {_stem_token(token) for token in _TOKEN_PATTERN.findall(value.casefold())}
            if term in tokens:
                best = max(best, weight)
            elif any(term in token for token in tokens):
                best = max(best, weight * 0.75)
        score += best
    return score / len(query_terms)


def _cosine_similarity(left: list[float], right: Any) -> float | None:
    if right is None or not hasattr(right, "__len__") or len(left) != len(right):
        return None
    try:
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
        right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    except (TypeError, ValueError, OverflowError):
        return None
    if not left_norm or not right_norm:
        return None
    value = dot / (left_norm * right_norm)
    return max(-1.0, min(1.0, value)) if math.isfinite(value) else None


def _match_from_values(
    chunk_id: Any,
    document: Any,
    metadata: Any,
    similarity: float,
) -> RetrievalMatch:
    if not isinstance(metadata, dict) or not isinstance(chunk_id, str) or not isinstance(document, str):
        raise RetrievalUnavailableError("Vector search returned an invalid result.")
    public_metadata = _safe_metadata(metadata)
    source_row_value = public_metadata.get("row_number")
    source_row = (
        source_row_value
        if isinstance(source_row_value, int) and not isinstance(source_row_value, bool)
        else None
    )
    return RetrievalMatch(
        rank=0,
        chunk_id=chunk_id,
        document=document,
        source_row=source_row,
        metadata=public_metadata,
        similarity_score=similarity,
    )


def retrieve_semantic_matches(
    upload_id: UUID | str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    vector_store: ChromaVectorStore | None = None,
    embedding_function: Callable[[str], list[float]] = generate_query_embedding,
) -> list[RetrievalMatch]:
    """Retrieve ranked chunks for exactly one indexed upload."""
    upload_id_value = str(upload_id).strip()
    if not upload_id_value:
        raise InvalidRetrievalRequestError("upload_id is required.")
    if not isinstance(query, str) or not query.strip():
        raise InvalidRetrievalRequestError("The retrieval query cannot be empty.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise InvalidRetrievalRequestError(
            f"top_k must be between 1 and {MAX_TOP_K}."
        )

    try:
        store = vector_store or ChromaVectorStore()
        stored_count = store.count(upload_id_value)
        if stored_count == 0:
            raise DatasetNotIndexedError("The selected dataset has not been indexed.")
        query_embedding = embedding_function(query.strip())
        result = store.query(
            upload_id_value,
            query_embedding,
            min(max(top_k, VECTOR_CANDIDATE_COUNT), stored_count),
        )
    except DatasetNotIndexedError:
        raise
    except (EmbeddingError, VectorStoreUnavailableError) as exc:
        raise RetrievalUnavailableError("Semantic retrieval is unavailable.") from exc

    ids = _first_query_results(result, "ids")
    documents = _first_query_results(result, "documents")
    metadatas = _first_query_results(result, "metadatas")
    distances = _first_query_results(result, "distances")
    if not len(ids) == len(documents) == len(metadatas) == len(distances):
        raise RetrievalUnavailableError("Vector search returned an invalid result.")

    matches = []
    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        if not isinstance(metadata, dict) or metadata.get("upload_id") != upload_id_value:
            raise RetrievalUnavailableError("Vector search returned an invalid result.")
        try:
            similarity = 1.0 - float(distance)
        except (TypeError, ValueError) as exc:
            raise RetrievalUnavailableError(
                "Vector search returned an invalid result."
            ) from exc
        if not math.isfinite(similarity):
            raise RetrievalUnavailableError("Vector search returned an invalid result.")
        similarity = max(-1.0, min(1.0, similarity))

        matches.append(_match_from_values(chunk_id, document, metadata, similarity))

    matches.sort(key=lambda match: match.similarity_score, reverse=True)
    vector_matches = {match.chunk_id: match for match in matches}
    query_terms = _query_terms(query)
    lexical_candidates: list[tuple[float, str, str, dict[str, Any]]] = []
    if query_terms and hasattr(store, "get_documents") and hasattr(store, "get_embeddings"):
        try:
            corpus = store.get_documents(upload_id_value)
            corpus_ids = corpus.get("ids") or []
            corpus_documents = corpus.get("documents") or []
            corpus_metadatas = corpus.get("metadatas") or []
            if not len(corpus_ids) == len(corpus_documents) == len(corpus_metadatas):
                raise VectorStoreUnavailableError("Stored searchable documents were invalid.")
            for chunk_id, document, metadata in zip(
                corpus_ids, corpus_documents, corpus_metadatas
            ):
                if (
                    not isinstance(chunk_id, str)
                    or not isinstance(document, str)
                    or not isinstance(metadata, dict)
                    or metadata.get("upload_id") != upload_id_value
                ):
                    continue
                lexical = _lexical_score(query_terms, metadata, document)
                if lexical > 0:
                    lexical_candidates.append((lexical, chunk_id, document, metadata))
            lexical_candidates.sort(
                key=lambda item: (
                    -item[0],
                    int(item[3].get("row_number", 0))
                    if isinstance(item[3].get("row_number"), int)
                    else 0,
                    item[1],
                )
            )
            lexical_candidates = lexical_candidates[:LEXICAL_CANDIDATE_COUNT]
            missing_ids = [
                item[1] for item in lexical_candidates if item[1] not in vector_matches
            ]
            stored_embeddings = store.get_embeddings(missing_ids)
            embedding_ids = stored_embeddings.get("ids") or []
            embedding_values = stored_embeddings.get("embeddings")
            if embedding_values is None:
                embedding_values = []
            embedding_by_id = dict(
                zip(
                    embedding_ids,
                    embedding_values,
                )
            )
            for _lexical, chunk_id, document, metadata in lexical_candidates:
                if chunk_id in vector_matches:
                    continue
                similarity = _cosine_similarity(query_embedding, embedding_by_id.get(chunk_id))
                if similarity is not None:
                    vector_matches[chunk_id] = _match_from_values(
                        chunk_id, document, metadata, similarity
                    )
        except VectorStoreUnavailableError:
            lexical_candidates = []

    lexical_scores = {item[1]: item[0] for item in lexical_candidates}
    if lexical_scores:
        matches = list(vector_matches.values())
        matches.sort(
            key=lambda match: (
                -(
                    LEXICAL_WEIGHT * lexical_scores.get(match.chunk_id, 0.0)
                    + VECTOR_WEIGHT * max(0.0, (match.similarity_score + 1.0) / 2.0)
                ),
                -match.similarity_score,
                match.source_row if match.source_row is not None else 0,
                match.chunk_id,
            )
        )
    else:
        matches = list(vector_matches.values())
        matches.sort(key=lambda match: match.similarity_score, reverse=True)
    return [
        RetrievalMatch(
            rank=rank,
            chunk_id=match.chunk_id,
            document=match.document,
            source_row=match.source_row,
            metadata=match.metadata,
            similarity_score=match.similarity_score,
        )
        for rank, match in enumerate(matches[:top_k], start=1)
    ]
