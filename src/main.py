"""
Main script for PDF parsing with CLI interface
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pdf_parser import DoclingPDFParser
from pdf_parser.utils import get_pdf_files, logger
from chunker import AnnualReportChunker
from embedding import (
    BGEEmbedder, MilvusVectorStore, HybridSearcher,
    EmbeddingConfig, MilvusConfig
)


def parse_single_file(args):
    """Parse a single PDF file."""
    parser = DoclingPDFParser(
        extract_tables=not args.no_tables,
        extract_images=not args.no_images,
        extract_structure=not args.no_structure,
        ocr_enabled=not args.no_ocr
    )

    # Parse and save as Markdown
    add_page_markers = not args.no_page_markers
    markdown_content = parser.save_to_markdown(args.input, args.output, add_page_markers=add_page_markers)

    # Print summary
    print(f"\n=== Markdown Export Summary ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Content Length: {len(markdown_content)} characters")

    if args.show_text:
        print(f"\n=== Markdown Preview (first 500 chars) ===")
        print(markdown_content[:500] + "...")


def parse_directory(args):
    """Parse all PDF files in a directory."""
    parser = DoclingPDFParser(
        extract_tables=not args.no_tables,
        extract_images=not args.no_images,
        extract_structure=not args.no_structure,
        ocr_enabled=not args.no_ocr
    )

    # Get all PDF files
    pdf_files = get_pdf_files(args.input)

    if not pdf_files:
        logger.warning(f"No PDF files found in {args.input}")
        return

    # Parse and save as Markdown
    add_page_markers = not args.no_page_markers
    parser.save_batch_to_markdown(
        pdf_files,
        args.output,
        merge=args.merge,
        add_page_markers=add_page_markers
    )

    # Print summary
    print(f"\n=== Batch Markdown Export Summary ===")
    print(f"Total Files: {len(pdf_files)}")
    print(f"Output Directory: {args.output}")
    print(f"Format: Markdown")
    if args.merge:
        print(f"Mode: Merged into single file")
    print(f"Page Markers: {'Enabled' if add_page_markers else 'Disabled'}")


def chunk_file(args):
    """Chunk a single Markdown file."""
    import json

    chunker = AnnualReportChunker()

    # Chunk the file
    content = Path(args.input).read_text(encoding='utf-8')
    sections = chunker.chunk_by_sections(
        content,
        min_chars=100,
        max_chars=args.chunk_size,
        merge_small=True
    )

    # Save chunks as JSON array
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{Path(args.input).stem}_chunks.json"

    # Convert chunks to the required format
    chunks_output = []
    for idx, section in enumerate(sections):
        chunk_data = {
            "chunk_id": f"chunk_{idx}",
            "content": section.get("content", ""),
            "metadata": {
                "source": str(args.input),
                "page": None,
                "section": section.get("title")
            }
        }
        chunks_output.append(chunk_data)

    # Write JSON array
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_output, f, ensure_ascii=False, indent=2)

    print(f"Saved chunks to: {output_file}")

    # Print summary
    print(f"\n=== Chunking Summary ===")
    print(f"Source: {args.input}")
    print(f"Output: {output_file}")
    print(f"Total Chunks: {len(sections)}")
    print(f"Chunk Size: {args.chunk_size} chars")
    print(f"Overlap: {args.overlap} chars")

    # Show chunk statistics
    sizes = [len(section.get("content", "")) for section in sections if section.get("content")]
    print(f"Size Statistics:")
    print(f"  Min: {min(sizes)} chars")
    print(f"  Max: {max(sizes)} chars")
    print(f"  Avg: {sum(sizes) // len(sizes)} chars")

    if args.show_chunks:
        print(f"\n=== First 3 Chunks Preview ===")
        for i, section in enumerate(sections[:3], 1):
            print(f"\n--- Chunk {i} ---")
            print(f"ID: chunk_{i - 1}")
            print(f"Section: {section.get('title', 'N/A')}")
            print(f"Page: N/A")
            print(f"Characters: {len(section.get('content', '').strip())}")
            print(f"Content Preview:\n{section.get('content', '')[:300]}...")


def chunk_directory(args):
    """Chunk all Markdown files in a directory."""
    import json

    chunker = AnnualReportChunker()

    # Find all Markdown files
    input_path = Path(args.input)
    md_files = list(input_path.glob("*.md")) + list(input_path.glob("*.markdown"))

    if not md_files:
        logger.warning(f"No Markdown files found in {args.input}")
        return

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_chunks = 0

    # Process each file
    for md_file in md_files:
        print(f"\nProcessing: {md_file.name}")
        content = md_file.read_text(encoding='utf-8')
        sections = chunker.chunk_by_sections(
            content,
            min_chars=100,
            max_chars=args.chunk_size,
            merge_small=True
        )
        total_chunks += len(sections)

        # Save chunks for this file as JSON
        output_file = output_path / f"{md_file.stem}_chunks.json"

        # Convert chunks to the required format
        chunks_output = []
        for idx, section in enumerate(sections):
            chunk_data = {
                "chunk_id": f"chunk_{idx}",
                "content": section.get("content", ""),
                "metadata": {
                    "source": str(md_file),
                    "page": None,
                    "section": section.get("title")
                }
            }
            chunks_output.append(chunk_data)

        # Write JSON array
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_output, f, ensure_ascii=False, indent=2)

        print(f"  -> {output_file.name} ({len(sections)} chunks)")

    # Print overall summary
    print(f"\n=== Batch Chunking Summary ===")
    print(f"Total Files: {len(md_files)}")
    print(f"Total Chunks: {total_chunks}")
    print(f"Output Directory: {output_path}")


def embed_json(args):
    """Embed chunks from JSON file and insert into Milvus."""
    # Initialize embedder
    embed_config = EmbeddingConfig(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size
    )
    embedder = BGEEmbedder(embed_config)

    # Initialize vector store
    store_config = MilvusConfig(
        host=args.host,
        port=args.port,
        collection_name=args.collection,
        index_type=args.index_type,
        metric_type=args.metric_type
    )
    store = MilvusVectorStore(store_config)

    # Create or load collection
    store.create_collection(overwrite=args.overwrite)
    store.load_collection()

    # Load and embed chunks
    count = store.insert_from_json(args.input, embedder)

    print(f"\n=== Embedding Summary ===")
    print(f"Input File: {args.input}")
    print(f"Collection: {args.collection}")
    print(f"Inserted Chunks: {count}")
    print(f"Model: {args.model}")
    print(f"Dimension: {embedder.get_dimension()}")

    # Print collection info
    info = store.get_collection_info()
    print(f"\nTotal Entities in Collection: {info['num_entities']}")


def search_vectors(args):
    """Search for similar vectors."""
    # Initialize embedder
    embed_config = EmbeddingConfig(
        model_name=args.model,
        device=args.device
    )
    embedder = BGEEmbedder(embed_config)

    # Initialize vector store
    store_config = MilvusConfig(
        host=args.host,
        port=args.port,
        collection_name=args.collection,
        metric_type=args.metric_type
    )
    store = MilvusVectorStore(store_config)
    store.load_collection()

    # Encode query
    query_embedding = embedder.encode_queries(args.query)

    # Search
    results = store.search(
        query_embedding=query_embedding[0],
        top_k=args.top_k
    )

    print(f"\n=== Search Results ===")
    print(f"Query: {args.query}")
    print(f"Top {args.top_k} Results:\n")

    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {result.source}")
        print(f"Page: {result.page}")
        print(f"Section: {result.section}")
        print(f"Content Preview: {result.content[:200]}...")
        print()


def hybrid_search_vectors(args):
    """Hybrid search combining BM25 and Vector retrieval."""
    import json

    # Initialize embedder
    embed_config = EmbeddingConfig(
        model_name=args.model,
        device=args.device
    )
    embedder = BGEEmbedder(embed_config)

    # Initialize vector store
    store_config = MilvusConfig(
        host=args.host,
        port=args.port,
        collection_name=args.collection,
        metric_type=args.metric_type
    )
    store = MilvusVectorStore(store_config)
    store.load_collection()

    # Initialize optional reranker
    reranker = None
    if args.rerank:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(args.rerank_model, device=args.device)
        except Exception as e:
            print(f"Warning: Failed to load reranker: {e}")

    # Initialize hybrid searcher
    hybrid_searcher = HybridSearcher(
        vector_store=store,
        embedder=embedder,
        reranker=reranker
    )

    # Load chunks for BM25
    if args.chunks_file:
        print(f"Loading chunks from {args.chunks_file} for BM25...")
        with open(args.chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        hybrid_searcher.load_chunks_for_bm25(chunks)
        print(f"Loaded {len(chunks)} chunks for BM25 indexing")
    else:
        print("Warning: No chunks file provided. BM25 will be skipped.")
        print("Use --chunks-file parameter to enable BM25 retrieval.")

    # Perform hybrid search
    print(f"\n=== Hybrid Search ===")
    print(f"Query: {args.query}")
    print(f"BM25 top-k: {args.bm25_k}")
    print(f"Vector top-k: {args.vector_k}")
    print(f"RRF k: {args.rrf_k}")
    print(f"RRF alpha (BM25 weight): {args.alpha}")
    print()

    results = hybrid_searcher.search_hybrid(
        query=args.query,
        top_k=args.top_k,
        bm25_k=args.bm25_k,
        vector_k=args.vector_k,
        rrf_k=args.rrf_k,
        alpha=args.alpha,
        rerank_top_k=args.rerank_top_k if args.rerank else None
    )

    print(f"\n=== Results (RRF Fusion) ===")
    print(f"Top {len(results)} Results:\n")

    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Fusion Score: {result['fusion_score']:.6f}")
        print(f"Vector Score: {result['vector_score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Section: {result['section']}")
        print(f"Content Preview: {result['content'][:200]}...")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Document Processor - Parse PDFs, chunk Markdown, and embed vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export single PDF to Markdown
  python main.py parse file.pdf -o output.md

  # Chunk a Markdown file
  python main.py chunk report.md -o chunks/

  # Embed chunks into Milvus
  python main.py embed chunks.json --overwrite

  # Search similar vectors
  python main.py search "茅台酒2024年营业收入"
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Parse command
    parse_parser = subparsers.add_parser('parse', help='Parse PDF file(s)')

    parse_parser.add_argument(
        'input',
        type=str,
        help='Path to PDF file or directory containing PDFs'
    )

    parse_parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file or directory'
    )

    parse_parser.add_argument(
        '--merge',
        action='store_true',
        help='Merge all results into a single file (only for directory input)'
    )

    parse_parser.add_argument(
        '--no-tables',
        action='store_true',
        help='Disable table extraction'
    )

    parse_parser.add_argument(
        '--no-images',
        action='store_true',
        help='Disable image extraction'
    )

    parse_parser.add_argument(
        '--no-structure',
        action='store_true',
        help='Disable document structure extraction'
    )

    parse_parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable OCR for scanned PDFs'
    )

    parse_parser.add_argument(
        '--show-text',
        action='store_true',
        help='Print extracted content preview to console'
    )

    parse_parser.add_argument(
        '--no-page-markers',
        action='store_true',
        help='Disable page number markers in Markdown output'
    )

    # Chunk command
    chunk_parser = subparsers.add_parser('chunk', help='Chunk Markdown file(s)')

    chunk_parser.add_argument(
        'input',
        type=str,
        help='Path to Markdown file or directory containing Markdown files'
    )

    chunk_parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output directory for chunks'
    )

    chunk_parser.add_argument(
        '--chunk-size',
        type=int,
        default=2000,
        help='Target chunk size in characters (default: 2000)'
    )

    chunk_parser.add_argument(
        '--overlap',
        type=int,
        default=200,
        help='Character overlap between chunks (default: 200)'
    )

    chunk_parser.add_argument(
        '--hard-limit',
        type=int,
        default=4000,
        help='Maximum allowed characters per chunk (default: 4000)'
    )

    chunk_parser.add_argument(
        '--no-table-protection',
        action='store_true',
        help='Disable table protection (allow splitting tables)'
    )

    chunk_parser.add_argument(
        '--show-chunks',
        action='store_true',
        help='Print preview of chunks to console'
    )

    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Embed chunks and insert into Milvus')

    embed_parser.add_argument(
        'input',
        type=str,
        help='Path to JSON file with chunks'
    )

    embed_parser.add_argument(
        '--model',
        type=str,
        default='BAAI/bge-m3',
        help='Embedding model name (default: BAAI/bge-m3)'
    )

    embed_parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device: cuda, cpu, or mps (default: cuda)'
    )

    embed_parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding (default: 32)'
    )

    embed_parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Milvus server host (default: localhost)'
    )

    embed_parser.add_argument(
        '--port',
        type=int,
        default=19530,
        help='Milvus server port (default: 19530)'
    )

    embed_parser.add_argument(
        '--collection',
        type=str,
        default='rag_chunks',
        help='Collection name (default: rag_chunks)'
    )

    embed_parser.add_argument(
        '--index-type',
        type=str,
        default='HNSW',
        choices=['FLAT', 'IVF_FLAT', 'HNSW'],
        help='Index type (default: HNSW)'
    )

    embed_parser.add_argument(
        '--metric-type',
        type=str,
        default='IP',
        choices=['IP', 'COSINE', 'L2'],
        help='Metric type (default: IP)'
    )

    embed_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing collection'
    )

    # Search command
    search_parser = subparsers.add_parser('search', help='Search for similar vectors')

    search_parser.add_argument(
        'query',
        type=str,
        help='Query text'
    )

    search_parser.add_argument(
        '--model',
        type=str,
        default='BAAI/bge-m3',
        help='Embedding model name (default: BAAI/bge-m3)'
    )

    search_parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device: cuda, cpu, or mps (default: cuda)'
    )

    search_parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Milvus server host (default: localhost)'
    )

    search_parser.add_argument(
        '--port',
        type=int,
        default=19530,
        help='Milvus server port (default: 19530)'
    )

    search_parser.add_argument(
        '--collection',
        type=str,
        default='rag_chunks',
        help='Collection name (default: rag_chunks)'
    )

    search_parser.add_argument(
        '--metric-type',
        type=str,
        default='IP',
        choices=['IP', 'COSINE', 'L2'],
        help='Metric type (default: IP)'
    )

    search_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )

    # Hybrid Search command
    hybrid_parser = subparsers.add_parser('hybrid-search', help='Hybrid search (BM25 + Vector + RRF)')

    hybrid_parser.add_argument(
        'query',
        type=str,
        help='Query text'
    )

    hybrid_parser.add_argument(
        '--chunks-file',
        type=str,
        required=True,
        help='Path to chunks JSON file for BM25 indexing'
    )

    hybrid_parser.add_argument(
        '--model',
        type=str,
        default='BAAI/bge-m3',
        help='Embedding model name (default: BAAI/bge-m3)'
    )

    hybrid_parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device: cuda, cpu, or mps (default: cuda)'
    )

    hybrid_parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Milvus server host (default: localhost)'
    )

    hybrid_parser.add_argument(
        '--port',
        type=int,
        default=19530,
        help='Milvus server port (default: 19530)'
    )

    hybrid_parser.add_argument(
        '--collection',
        type=str,
        default='rag_chunks',
        help='Collection name (default: rag_chunks)'
    )

    hybrid_parser.add_argument(
        '--metric-type',
        type=str,
        default='IP',
        choices=['IP', 'COSINE', 'L2'],
        help='Metric type (default: IP)'
    )

    hybrid_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Final number of results to return (default: 5)'
    )

    hybrid_parser.add_argument(
        '--bm25-k',
        type=int,
        default=50,
        help='Number of BM25 results for fusion (default: 50)'
    )

    hybrid_parser.add_argument(
        '--vector-k',
        type=int,
        default=50,
        help='Number of Vector results for fusion (default: 50)'
    )

    hybrid_parser.add_argument(
        '--rrf-k',
        type=int,
        default=60,
        help='RRF constant k (default: 60)'
    )
    hybrid_parser.add_argument(
        '--rerank',
        action='store_true',
        help='Enable CrossEncoder rerank (default: False)'
    )
    hybrid_parser.add_argument(
        '--rerank-model',
        type=str,
        default='BAAI/bge-reranker-large',
        help='CrossEncoder model name (default: BAAI/bge-reranker-large)'
    )
    hybrid_parser.add_argument(
        '--rerank-top-k',
        type=int,
        default=10,
        help='Rerank top-k candidates (default: 10)'
    )
    hybrid_parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Weight for BM25 in RRF fusion (default: 0.5)'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if args.command == 'parse':
        input_path = Path(args.input)

        if input_path.is_file():
            parse_single_file(args)
        elif input_path.is_dir():
            parse_directory(args)
        else:
            logger.error(f"Invalid path: {args.input}")
            sys.exit(1)

    elif args.command == 'chunk':
        input_path = Path(args.input)

        if input_path.is_file():
            chunk_file(args)
        elif input_path.is_dir():
            chunk_directory(args)
        else:
            logger.error(f"Invalid path: {args.input}")
            sys.exit(1)

    elif args.command == 'embed':
        embed_json(args)

    elif args.command == 'search':
        search_vectors(args)

    elif args.command == 'hybrid-search':
        hybrid_search_vectors(args)


if __name__ == "__main__":
    main()
