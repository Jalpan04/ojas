"""File walker and ChromaDB ingestion for the local codebase.

Walks the workspace directory, chunks each indexable file, and upserts the
chunks into a persistent ChromaDB collection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from rich.console import Console
# No longer using rich.progress to avoid background thread blocking the terminal.

from ojas.rag.embeddings import get_embeddings
from ojas.utils import EXCLUDED_DIRS, chunk_text, is_indexable

console = Console()

COLLECTION_NAME = "ojas_workspace"
INDEXING_COMPLETE = False
IS_INDEXING = False


def _file_id(path: Path, chunk_index: int) -> str:
    """Deterministic document ID for a file chunk."""
    return hashlib.sha256(
        f"{path.as_posix()}::{chunk_index}".encode()
    ).hexdigest()[:24]


def index_workspace(
    workspace: str | Path,
    embed_model: str = "nomic-embed-text",
    ollama_url: str = "http://localhost:11434",
    db_path: str | None = None,
) -> chromadb.Collection:
    """Walk *workspace*, chunk files, and upsert into ChromaDB."""
    global INDEXING_COMPLETE, IS_INDEXING
    IS_INDEXING = True
    workspace = Path(workspace).resolve()
    if db_path is None:
        db_path = str(workspace / ".ojas" / "chromadb")

    client = chromadb.PersistentClient(path=db_path)

    # Use a fresh collection each run so stale files are cleaned up.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # Collection may not exist on first run (NotFoundError in newer
        # ChromaDB versions, ValueError in older ones).
        pass

    embedder = get_embeddings(model=embed_model, base_url=ollama_url)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    files = [
        p
        for p in workspace.rglob("*")
        if p.is_file() and is_indexable(p)
    ]

    if not files:
        console.print("  [yellow]No indexable files found.[/]")
        return collection

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    texts_to_embed: list[str] = []

    for fpath in files:
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue

        relative = fpath.relative_to(workspace).as_posix()
        chunks = chunk_text(content)

        for idx, chunk in enumerate(chunks):
            doc_id = _file_id(fpath, idx)
            documents.append(chunk)
            metadatas.append(
                {
                    "source": relative,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }
            )
            ids.append(doc_id)
            texts_to_embed.append(chunk)

    if not documents:
        console.print("  [yellow]No content to index.[/]")
        return collection

    # Embed in batches to avoid overwhelming Ollama.
    BATCH = 32
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts_to_embed), BATCH):
        batch = texts_to_embed[i : i + BATCH]
        batch_emb = embedder.embed_documents(batch)
        all_embeddings.extend(batch_emb)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=all_embeddings,
    )

    INDEXING_COMPLETE = True
    IS_INDEXING = False

    console.print(
        f"\r[green]✓ Indexed {len(files)} files "
        f"({len(documents)} chunks) into ChromaDB.[/]"
    )
    return collection


def start_background_indexing(
    workspace: str | Path,
    embed_model: str = "nomic-embed-text",
    ollama_url: str = "http://localhost:11434",
    db_path: str | None = None,
):
    """Start index_workspace in a background thread."""
    import threading

    thread = threading.Thread(
        target=index_workspace,
        args=(workspace, embed_model, ollama_url, db_path),
        daemon=True,
    )
    thread.start()
    return thread


def get_collection(
    workspace: str | Path,
    db_path: str | None = None,
) -> chromadb.Collection:
    """Get a handle to the workspace collection."""
    workspace = Path(workspace).resolve()
    if db_path is None:
        db_path = str(workspace / ".ojas" / "chromadb")

    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
