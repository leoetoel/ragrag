"""
Core PDF Parser using Docling - Simplified Version

This version focuses on text extraction with header/footer filtering.
"""

from pathlib import Path
from typing import Union, List, Optional
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, FormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

from .models import ParseResult, TextContent
from .utils import validate_pdf_path, logger


class DoclingPDFParser:
    """
    PDF Parser using Docling library with header/footer filtering.

    This class provides methods to parse PDF documents and extract text,
    with intelligent header/footer filtering based on position and patterns.
    """

    def __init__(
        self,
        ocr_enabled: bool = True,
        exclude_headers_footers: bool = False,
        header_margin_ratio: float = 0.15,
        footer_margin_ratio: float = 0.1,
        header_footer_patterns: bool = True,
        custom_header_patterns: list = None,
        custom_footer_patterns: list = None,
        extract_tables: bool = True,
        extract_images: bool = True,
        extract_structure: bool = True
    ):
        """
        Initialize the PDF parser.

        Args:
            ocr_enabled: Whether to enable OCR for extracting text from images
            exclude_headers_footers: Whether to exclude headers and footers
            header_margin_ratio: Ratio of page height to consider as header area (default: 0.15)
            footer_margin_ratio: Ratio of page height to consider as footer area (default: 0.1)
            header_footer_patterns: Whether to use pattern matching for better detection
            custom_header_patterns: Custom regex patterns for headers
            custom_footer_patterns: Custom regex patterns for footers
            extract_tables: Whether to extract tables (kept for compatibility)
            extract_images: Whether to extract images (kept for compatibility)
            extract_structure: Whether to extract document structure (kept for compatibility)
        """
        self.ocr_enabled = ocr_enabled
        self.exclude_headers_footers = exclude_headers_footers
        self.header_margin_ratio = header_margin_ratio
        self.footer_margin_ratio = footer_margin_ratio
        self.header_footer_patterns = header_footer_patterns

        # Initialize default patterns
        self.header_patterns = self._get_default_header_patterns()
        self.footer_patterns = self._get_default_footer_patterns()

        # Add custom patterns if provided
        if custom_header_patterns:
            import re
            self.header_patterns.extend([re.compile(p, re.IGNORECASE) for p in custom_header_patterns])
        if custom_footer_patterns:
            import re
            self.footer_patterns.extend([re.compile(p, re.IGNORECASE) for p in custom_footer_patterns])

        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ocr_enabled
        pipeline_options.do_table_structure = False
        pipeline_options.do_picture_description = False

        # Initialize document converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: FormatOption(
                    backend=PyPdfiumDocumentBackend,
                    pipeline_cls=StandardPdfPipeline,
                    pipeline_options=pipeline_options
                )
            }
        )

        logger.info(f"DoclingPDFParser initialized (ocr={ocr_enabled}, "
                   f"exclude_headers_footers={exclude_headers_footers})")

    def _get_default_header_patterns(self) -> list:
        """Get default regex patterns for identifying header text."""
        import re
        patterns = [
            # Year + report type
            r'\d{4}\s*年\s*度\s*报\s*告',
            r'\d{4}\s*年\s*中\s*报',
            r'\d{4}\s*Annual\s*Report',

            # Dates (very common in headers)
            r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]',
            r'\d{4}[-/]\d{1,2}[-/]',
            r'\d{4}[-/]\d{1,2}$',

            # Company info
            r'公\s*司\s*代\s*码\s*：\s*\d+',
            r'公\s*司\s*简\s*称\s*：\s*[\u4e00-\u9fff]+',
            r'代\s*码\s*：\s*\d{6}',

            # Stock codes
            r'\d{6}',

            # Disclaimer/Statement text
            r'重\s*要\s*提\s*示',
            r'特\s*别\s*说\s*明',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _get_default_footer_patterns(self) -> list:
        """Get default regex patterns for identifying footer text."""
        import re
        patterns = [
            # Page numbers
            r'-\s*\d+\s*-',
            r'第\s*\d+\s*页',
            r'Page\s*\d+',
            r'\d+\s*/\s*\d+',

            # Common footer text
            r'机\s*密',
            r'内\s*部\s*资\s*料',
            r'协[\u3000 ]+会',
            r'SEMI',

            # Date stamps
            r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def parse_pdf(self, pdf_path: Union[str, Path]) -> ParseResult:
        """
        Parse a single PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ParseResult object containing all extracted data

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file is not a valid PDF
            Exception: For parsing errors
        """
        # Validate path
        path = validate_pdf_path(pdf_path)

        logger.info(f"Parsing PDF: {path}")

        try:
            # Convert document
            doc = self.converter.convert(str(path))

            # Initialize result
            result = ParseResult(
                file_path=str(path),
                file_name=path.name,
                total_pages=len(doc.pages),
                metadata=self._extract_metadata(doc)
            )

            # Extract full text as markdown
            result.full_text = doc.document.export_to_markdown()

            # Extract text content
            self._extract_text(doc.document, result)

            logger.info(f"Successfully parsed {path.name}: "
                       f"{len(result.text_content)} text blocks")

            return result

        except Exception as e:
            logger.error(f"Error parsing PDF {path}: {str(e)}")
            raise

    def _extract_metadata(self, doc) -> dict:
        """Extract document metadata."""
        metadata = {}

        if hasattr(doc, 'document') and hasattr(doc.document, 'meta'):
            meta = doc.document.meta
            for key, value in meta.items():
                metadata[key] = str(value) if value else None

        return metadata

    def _extract_text(self, doc_content, result: ParseResult) -> None:
        """Extract all text content with page information."""
        try:
            # Track page sizes for filtering
            page_sizes_seen = {}

            # Iterate through text elements
            for item in doc_content.texts:
                page_num = 1
                bbox = None
                page_idx = 0

                # Get page number and bbox from provenance
                if hasattr(item, 'prov') and len(item.prov) > 0:
                    prov = item.prov[0]
                    page_idx = prov.page_no
                    page_num = page_idx + 1
                    if hasattr(prov, 'bbox'):
                        bbox = list(prov.bbox.as_tuple())

                    # Get page size for filtering
                    if self.exclude_headers_footers and page_idx not in page_sizes_seen:
                        if hasattr(doc_content, 'pages') and page_idx in doc_content.pages:
                            page_obj = doc_content.pages[page_idx]
                            if hasattr(page_obj, 'size'):
                                page_sizes_seen[page_idx] = page_obj.size

                # Skip headers and footers if enabled
                if self.exclude_headers_footers and bbox and page_idx in page_sizes_seen:
                    if self._is_header_or_footer(bbox, page_sizes_seen[page_idx], item.text):
                        continue

                text_content = TextContent(
                    text=item.text,
                    page_num=page_num,
                    bbox=bbox
                )
                result.text_content.append(text_content)

        except Exception as e:
            logger.warning(f"Error extracting text: {e}")

    def _is_header_or_footer(self, bbox: List[float], page_size, text: str) -> bool:
        """
        Check if text element is in header or footer area.

        Args:
            bbox: Bounding box as [left, bottom, right, top]
            page_size: Page size object with width and height attributes
            text: The text content to check for patterns

        Returns:
            True if the element should be excluded
        """
        if not bbox or len(bbox) < 4:
            return False

        # Get page dimensions
        page_height = getattr(page_size, 'height', None)
        if page_height is None or page_height <= 0:
            return False

        # bbox is [left, bottom, right, top]
        text_bottom = bbox[1]
        text_top = bbox[3]

        # Calculate thresholds
        header_threshold = page_height * (1 - self.header_margin_ratio)
        footer_threshold = page_height * self.footer_margin_ratio

        # Check position
        in_header_area = text_top > header_threshold
        in_footer_area = text_bottom < footer_threshold

        if not (in_header_area or in_footer_area):
            return False

        # If pattern matching disabled, filter by position only
        if not self.header_footer_patterns:
            return True

        # Check patterns
        if in_header_area:
            return self._matches_header_pattern(text)

        if in_footer_area:
            return self._matches_footer_pattern(text)

        return False

    def _matches_header_pattern(self, text: str) -> bool:
        """Check if text matches header patterns."""
        if not text:
            return False

        text_clean = text.strip()
        if len(text_clean) <= 2:
            return False

        for pattern in self.header_patterns:
            if pattern.search(text_clean):
                return True

        return False

    def _matches_footer_pattern(self, text: str) -> bool:
        """Check if text matches footer patterns."""
        if not text:
            return False

        text_clean = text.strip()
        if len(text_clean) <= 1:
            return False

        for pattern in self.footer_patterns:
            if pattern.search(text_clean):
                return True

        return False

    def save_to_markdown(
        self,
        pdf_path: Union[str, Path],
        output_path: Union[str, Path],
        add_page_markers: bool = True
    ) -> str:
        """
        Parse PDF and save as Markdown file.

        Args:
            pdf_path: Path to the PDF file
            output_path: Path to save the Markdown file
            add_page_markers: Whether to add page number markers

        Returns:
            The markdown content as string
        """
        # Validate path
        path = validate_pdf_path(pdf_path)

        logger.info(f"Parsing PDF to Markdown: {path}")

        try:
            # Convert document
            doc = self.converter.convert(str(path))

            if add_page_markers:
                # Export with page markers
                markdown_content = self._export_to_markdown_with_pages(doc)
            else:
                # Standard export
                markdown_content = doc.document.export_to_markdown()

            # Save to file
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            with open(output, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            logger.info(f"Saved Markdown to {output} ({len(markdown_content)} chars)")

            return markdown_content

        except Exception as e:
            logger.error(f"Error parsing PDF to Markdown {path}: {str(e)}")
            raise

    def save_batch_to_markdown(
        self,
        pdf_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        merge: bool = False,
        add_page_markers: bool = True
    ) -> None:
        """
        Parse batch PDFs and save as Markdown files.

        Args:
            pdf_paths: List of paths to PDF files
            output_dir: Directory to save Markdown files
            merge: Whether to merge all results into one file
            add_page_markers: Whether to add page number markers
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if merge:
            # Merge all markdown into one file
            all_markdowns = []

            for pdf_path in pdf_paths:
                try:
                    logger.info(f"Processing: {pdf_path}")
                    doc = self.converter.convert(str(pdf_path))

                    # Export with or without page markers
                    if add_page_markers:
                        markdown_content = self._export_to_markdown_with_pages(doc)
                    else:
                        markdown_content = doc.document.export_to_markdown()

                    # Add file name as header
                    pdf_name = Path(pdf_path).stem
                    section = f"\n\n# {pdf_name}\n\n{markdown_content}\n\n---\n"
                    all_markdowns.append(section)

                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {str(e)}")

            # Write merged file
            merged_file = output_path / "merged_results.md"
            with open(merged_file, 'w', encoding='utf-8') as f:
                f.write("".join(all_markdowns))

            logger.info(f"Merged {len(all_markdowns)} files to {merged_file}")
        else:
            # Save each PDF to separate markdown file
            for pdf_path in pdf_paths:
                try:
                    pdf_name = Path(pdf_path).stem
                    output_file = output_path / f"{pdf_name}.md"
                    self.save_to_markdown(pdf_path, output_file, add_page_markers)
                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {str(e)}")

    def _export_to_markdown_with_pages(self, doc) -> str:
        """
        Export document to Markdown with page number markers.

        This implementation properly handles texts, tables, and pictures by
        sorting all elements by their page number and position.

        Args:
            doc: Docling document object

        Returns:
            Markdown content with page markers
        """
        result_parts = []
        current_page = 0

        # Create a list of all elements with their page info and export function
        elements = []

        # Add text elements
        for text_item in doc.document.texts:
            page_no = 1
            bbox = None

            if hasattr(text_item, 'prov') and len(text_item.prov) > 0:
                page_no = text_item.prov[0].page_no + 1
                if hasattr(text_item.prov[0], 'bbox'):
                    bbox = text_item.prov[0].bbox

            elements.append({
                'type': 'text',
                'page_no': page_no,
                'bbox': bbox,
                'content': text_item.text if hasattr(text_item, 'text') else str(text_item)
            })

        # Add table elements
        for table_item in doc.document.tables:
            page_no = 1
            bbox = None
            if hasattr(table_item, 'prov') and len(table_item.prov) > 0:
                page_no = table_item.prov[0].page_no + 1
                if hasattr(table_item.prov[0], 'bbox'):
                    bbox = table_item.prov[0].bbox

            # Export table to markdown
            table_md = table_item.export_to_markdown(doc=doc.document) if hasattr(table_item, 'export_to_markdown') else ''

            elements.append({
                'type': 'table',
                'page_no': page_no,
                'bbox': bbox,
                'content': table_md
            })

        # Add picture elements
        for picture_item in doc.document.pictures:
            page_no = 1
            bbox = None
            if hasattr(picture_item, 'prov') and len(picture_item.prov) > 0:
                page_no = picture_item.prov[0].page_no + 1
                if hasattr(picture_item.prov[0], 'bbox'):
                    bbox = picture_item.prov[0].bbox

            # Export picture to markdown
            pic_md = picture_item.export_to_markdown(doc=doc.document) if hasattr(picture_item, 'export_to_markdown') else '<!-- image -->'

            elements.append({
                'type': 'picture',
                'page_no': page_no,
                'bbox': bbox,
                'content': pic_md
            })

        # Sort elements by page number and then by bbox (top to bottom, left to right)
        def get_sort_key(element):
            key = [element['page_no']]
            if element['bbox'] is not None:
                # bbox is (left, top, right, bottom) - we want top (y) first, then left (x)
                key.extend([element['bbox'].t, element['bbox'].l])
            else:
                key.extend([0, 0])
            return tuple(key)

        elements.sort(key=get_sort_key)

        # Build result with page markers
        for element in elements:
            page_no = element['page_no']
            content = element['content']

            # Add page marker if page changed
            if page_no != current_page:
                if result_parts:  # Not the first element
                    result_parts.append("\n\n---\n\n")
                result_parts.append(f"**第 {page_no} 页**\n\n")
                current_page = page_no

            # Add content
            result_parts.append(content)

            # Add newline between elements of the same page
            if element['type'] != 'picture':
                result_parts.append('\n')
            # Extra newline after tables for readability
            if element['type'] == 'table':
                result_parts.append('\n')

        return ''.join(result_parts)
