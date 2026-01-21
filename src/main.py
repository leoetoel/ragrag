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


def parse_single_file(args):
    """Parse a single PDF file."""
    parser = DoclingPDFParser(
        extract_tables=not args.no_tables,
        extract_images=not args.no_images,
        extract_structure=not args.no_structure,
        ocr_enabled=args.ocr
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
        ocr_enabled=args.ocr
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PDF Parser using Docling - Export PDFs to Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export single PDF to Markdown
  python main.py parse file.pdf -o output.md

  # Export all PDFs in directory to Markdown
  python main.py parse data/ -o output/

  # Merge all PDFs into one Markdown file
  python main.py parse data/ -o output/ --merge

  # Show Markdown preview
  python main.py parse file.pdf -o output.md --show-text
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
        '--ocr',
        action='store_true',
        help='Enable OCR for scanned PDFs'
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


if __name__ == "__main__":
    main()
