"""File walker and ChromaDB ingestion for the local codebase.

Walks the workspace directory, chunks each indexable file, and upserts the
chunks into a persistent ChromaDB collection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ojas.rag.embeddings import get_embeddings
from ojas.utils import EXCLUDED_DIRS, chunk_text, is_indexable

console = Console()

COLLECTION_NAME = "ojas_workspace"


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
    """Walk *workspace*, chunk files, and upsert into ChromaDB.

    Parameters
    ----------
    workspace:
        Root directory to index.
    embed_model:
        Ollama embedding model name.
    ollama_url:
        Ollama base URL.
    db_path:
        Optional path for the persistent ChromaDB store.  Defaults to
        ``<workspace>/.ojas/chromadb``.

    Returns
    -------
    chromadb.Collection
        The populated ChromaDB collection handle.
    """
    workspace = Path(workspace).resolve()
    if db_path is None:
        db_path = str(workspace / ".ojas" / "chromadb")

    client = chromadb.PersistentClient(path=db_path)

    # Use a fresh collection each run so stale files are cleaned up.
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Indexing workspace..."),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Indexing", total=len(files))
        for fpath in files:
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                progress.advance(task)
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

            progress.advance(task)

    if not documents:
        console.print("  [yellow]No content to index.[/]")
        return collection

    # Embed in batches to avoid overwhelming Ollama.
    BATCH = 32
    all_embeddings: list[list[float]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Generating embeddings..."),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Embedding", total=len(texts_to_embed))
        for i in range(0, len(texts_to_embed), BATCH):
            batch = texts_to_embed[i : i + BATCH]
            batch_emb = embedder.embed_documents(batch)
            all_embeddings.extend(batch_emb)
            progress.advance(task, advance=len(batch))

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=all_embeddings,
    )

    console.print(
        f"  [green]Indexed {len(files)} files "
        f"({len(documents)} chunks) into ChromaDB.[/]"
    )
    return collection
