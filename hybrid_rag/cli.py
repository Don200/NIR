"""CLI for Hybrid RAG indexing and management."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file
load_dotenv()

from .core.config import Config, load_config
from .core.models import Document
from .indexing.vector_index import VectorIndex
from .indexing.graph_index import GraphIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_documents_from_dir(path: Path) -> list[Document]:
    """Load documents from directory."""
    documents = []

    # Support .txt, .md, .json files
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                if file_path.suffix == ".json":
                    # JSON format: {"id": ..., "content": ..., "title": ...}
                    with open(file_path, encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                documents.append(Document(
                                    id=item.get("id", file_path.stem),
                                    content=item.get("content", item.get("text", "")),
                                    title=item.get("title", ""),
                                    metadata=item.get("metadata", {}),
                                ))
                        else:
                            documents.append(Document(
                                id=data.get("id", file_path.stem),
                                content=data.get("content", data.get("text", "")),
                                title=data.get("title", ""),
                                metadata=data.get("metadata", {}),
                            ))
                elif file_path.suffix in (".txt", ".md"):
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()
                    documents.append(Document(
                        id=file_path.stem,
                        content=content,
                        title=file_path.name,
                    ))
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

    return documents


def cmd_index(args: argparse.Namespace) -> int:
    """Index documents."""
    config = load_config(Path(args.config) if args.config else None)

    # Override output dir if specified
    if args.output:
        config.index_dir = Path(args.output)

    # Load documents
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1

    documents = load_documents_from_dir(input_path)
    if not documents:
        logger.error("No documents found")
        return 1

    logger.info(f"Loaded {len(documents)} documents")

    # Index
    if args.type in ("vector", "all"):
        logger.info("Building vector index...")
        vector_index = VectorIndex(config)
        count = vector_index.index_documents(documents)
        logger.info(f"Vector index: {count} chunks indexed")

    if args.type in ("graph", "all"):
        logger.info("Building graph index...")
        graph_index = GraphIndex(config)
        count = graph_index.index_documents(documents)
        logger.info(f"Graph index: {count} triplets extracted")

    # Save metadata
    metadata = {
        "document_count": len(documents),
        "index_types": args.type,
    }
    metadata_path = config.index_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Indexes saved to {config.index_dir}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Check index status."""
    config = load_config(Path(args.config) if args.config else None)

    if args.index_dir:
        config.index_dir = Path(args.index_dir)

    vector_index = VectorIndex(config)
    graph_index = GraphIndex(config)

    print(f"Index directory: {config.index_dir}")
    print(f"Vector index: {'Ready' if vector_index.is_indexed() else 'Not found'}")
    if vector_index.is_indexed():
        print(f"  Chunks: {vector_index.collection.count()}")

    print(f"Graph index: {'Ready' if graph_index.is_indexed() else 'Not found'}")

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start API server."""
    import uvicorn
    from .api.app import create_app

    config = load_config(Path(args.config) if args.config else None)

    if args.index_dir:
        config.index_dir = Path(args.index_dir)

    app = create_app(config)

    uvicorn.run(
        app,
        host=args.host or config.api_host,
        port=args.port or config.api_port,
    )
    return 0


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="hybrid_rag",
        description="Hybrid RAG - Vector + Graph retrieval",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--config", "-c", help="Path to config YAML file")
    index_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to documents directory",
    )
    index_parser.add_argument(
        "--output", "-o",
        help="Output directory for indexes",
    )
    index_parser.add_argument(
        "--type", "-t",
        choices=["vector", "graph", "all"],
        default="all",
        help="Index type to build",
    )

    status_parser = subparsers.add_parser("status", help="Check index status")
    status_parser.add_argument("--config", "-c", help="Path to config YAML file")
    status_parser.add_argument("--index-dir", help="Index directory")

    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--config", "-c", help="Path to config YAML file")
    serve_parser.add_argument("--host", help="Host to bind")
    serve_parser.add_argument("--port", type=int, help="Port to bind")
    serve_parser.add_argument("--index-dir", help="Index directory")

    args = parser.parse_args()

    commands = {
        "index": cmd_index,
        "status": cmd_status,
        "serve": cmd_serve,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
