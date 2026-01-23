"""
Core PDF Parser using Docling
"""

from pathlib import Path
from collections import defaultdict
from typing import Union, List, Optional, Dict, Any
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, FormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.document import TableCell as DoclingTableCell

from .models import (
    ParseResult,
    TextContent,
    TableInfo,
    TableCell,
    ImageInfo,
    DocumentStructure
)
from .utils import validate_pdf_path, logger


class DoclingPDFParser:
    """
    PDF Parser using Docling library.

    This class provides methods to parse PDF documents and extract
    text, tables, images, and document structure.
    """

    def __init__(
        self,
        extract_tables: bool = True,
        extract_images: bool = True,
        extract_structure: bool = True,
        ocr_enabled: bool = True,  # Enable OCR by default to extract text from images
        exclude_headers_footers: bool = False,  # Exclude headers and footers
        header_margin_ratio: float = 0.15,  # Top 15% of page is header (increased for better coverage)
        footer_margin_ratio: float = 0.1,  # Bottom 10% of page is footer
        header_footer_patterns: bool = True,  # Use pattern matching to identify headers/footers
        custom_header_patterns: list = None,  # Custom regex patterns for headers
        custom_footer_patterns: list = None,  # Custom regex patterns for footers
        auto_header_footer: bool = True,  # Detect repeated header/footer text by position
        min_header_footer_repeat_ratio: float = 0.3,  # Min ratio of pages for repeat detection
        min_header_footer_repeat_pages: int = 2  # Min pages for repeat detection
    ):
        """
        Initialize the PDF parser.

        Args:
            extract_tables: Whether to extract tables
            extract_images: Whether to extract images
            extract_structure: Whether to extract document structure
            ocr_enabled: Whether to enable OCR for extracting text from images (default: True)
            exclude_headers_footers: Whether to exclude headers and footers (default: False)
            header_margin_ratio: Ratio of page height to consider as header area (default: 0.15 = 15%)
            footer_margin_ratio: Ratio of page height to consider as footer area (default: 0.1 = 10%)
            header_footer_patterns: Whether to use pattern matching for better detection (default: True)
            custom_header_patterns: Custom regex patterns to identify header text
            custom_footer_patterns: Custom regex patterns to identify footer text
            auto_header_footer: Whether to auto-detect repeated header/footer text (default: True)
            min_header_footer_repeat_ratio: Min ratio of pages to treat as repeated (default: 0.3)
            min_header_footer_repeat_pages: Min pages to treat as repeated (default: 2)
        """
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.extract_structure = extract_structure
        self.ocr_enabled = ocr_enabled
        self.exclude_headers_footers = exclude_headers_footers
        self.header_margin_ratio = header_margin_ratio
        self.footer_margin_ratio = footer_margin_ratio
        self.header_footer_patterns = header_footer_patterns
        self.auto_header_footer = auto_header_footer
        self.min_header_footer_repeat_ratio = min_header_footer_repeat_ratio
        self.min_header_footer_repeat_pages = min_header_footer_repeat_pages

        # Initialize default patterns
        self.header_patterns = self._get_default_header_patterns()
        self.footer_patterns = self._get_default_footer_patterns()

        # Add custom patterns if provided
        if custom_header_patterns:
            self.header_patterns.extend(custom_header_patterns)
        if custom_footer_patterns:
            self.footer_patterns.extend(custom_footer_patterns)

        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options.device = "cuda"
        pipeline_options.do_ocr = ocr_enabled
        pipeline_options.do_table_structure = extract_tables

        # Disable picture description to save time (no VLM model needed)
        pipeline_options.do_picture_description = False

        # Initialize document converter with pipeline options
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: FormatOption(
                    backend=PyPdfiumDocumentBackend,
                    pipeline_cls=StandardPdfPipeline,
                    pipeline_options=pipeline_options
                )
            }
        )

        logger.info(f"DoclingPDFParser initialized (tables={extract_tables}, "
                   f"images={extract_images}, structure={extract_structure}, ocr={ocr_enabled})")

    def _get_markdown_content(self, doc, add_page_markers: bool) -> str:
        """Return markdown content from a Docling document."""
        if add_page_markers:
            return self._export_to_markdown_with_pages(doc)
        return doc.document.export_to_markdown()

    def _save_markdown(self, output_path: Union[str, Path], content: str) -> None:
        """Write markdown content to disk."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            f.write(content)

    def _get_prov_info(self, item) -> tuple:
        """
        Extract provenance info from a Docling item.

        Returns:
            (page_num, bbox, page_idx)
        """
        page_num = 1
        bbox = None
        page_idx = 0
        if hasattr(item, 'prov') and len(item.prov) > 0:
            prov = item.prov[0]
            page_idx = prov.page_no
            page_num = page_idx + 1
            if hasattr(prov, 'bbox'):
                bbox = prov.bbox
        return page_num, bbox, page_idx

    def _get_page_sizes(self, doc_content) -> Dict[int, Any]:
        """Return mapping of page index to size if available."""
        page_sizes = {}
        if hasattr(doc_content, 'pages'):
            try:
                for page_idx, page_obj in doc_content.pages.items():
                    if hasattr(page_obj, 'size'):
                        page_sizes[page_idx] = page_obj.size
            except Exception:
                pass
        return page_sizes

    def _normalize_hf_text(self, text: str) -> str:
        """Normalize header/footer text for repeat detection."""
        import re
        if not text:
            return ""
        text_clean = text.strip().lower()
        return re.sub(r'\s+', ' ', text_clean)

    def _get_header_footer_area_flags(self, bbox: List[float], page_size) -> tuple:
        """Return (in_header_area, in_footer_area) based on bbox and page size."""
        if not bbox or len(bbox) < 4:
            return False, False

        page_height = getattr(page_size, 'height', None)
        if page_height is None and isinstance(page_size, (list, tuple)) and len(page_size) >= 2:
            page_height = page_size[1]

        if page_height is None or page_height <= 0:
            return False, False

        text_bottom = bbox[1]
        text_top = bbox[3]

        header_threshold = page_height * (1 - self.header_margin_ratio)
        footer_threshold = page_height * self.footer_margin_ratio

        in_header_area = text_top > header_threshold
        in_footer_area = text_bottom < footer_threshold
        return in_header_area, in_footer_area

    def _build_repeated_header_footer_texts(self, doc_content, page_sizes: Dict[int, Any]) -> Dict[str, set]:
        """Build repeated header/footer text sets based on position across pages."""
        if not page_sizes:
            return {"header": set(), "footer": set()}

        header_hits = defaultdict(set)
        footer_hits = defaultdict(set)
        max_page_idx = -1

        for item in doc_content.texts:
            _, bbox_obj, page_idx = self._get_prov_info(item)
            if page_idx > max_page_idx:
                max_page_idx = page_idx
            if page_idx not in page_sizes:
                continue
            if bbox_obj is None:
                continue

            text_raw = item.text if hasattr(item, 'text') else str(item)
            text_norm = self._normalize_hf_text(text_raw)
            if len(text_norm) <= 2:
                continue

            bbox_list = list(bbox_obj.as_tuple()) if hasattr(bbox_obj, 'as_tuple') else list(bbox_obj)
            in_header_area, in_footer_area = self._get_header_footer_area_flags(
                bbox_list,
                page_sizes[page_idx]
            )

            if in_header_area:
                header_hits[text_norm].add(page_idx)
            if in_footer_area:
                footer_hits[text_norm].add(page_idx)

        page_count = len(page_sizes) if page_sizes else max_page_idx + 1
        if page_count <= 0:
            return {"header": set(), "footer": set()}

        min_pages = max(
            self.min_header_footer_repeat_pages,
            int(page_count * self.min_header_footer_repeat_ratio)
        )
        if min_pages <= 1:
            min_pages = 2

        header_texts = {text for text, pages in header_hits.items() if len(pages) >= min_pages}
        footer_texts = {text for text, pages in footer_hits.items() if len(pages) >= min_pages}
        return {"header": header_texts, "footer": footer_texts}

    def _get_default_header_patterns(self) -> list:
        """
        Get default regex patterns for identifying header text.

        Returns:
            List of regex patterns
        """
        import re
        patterns = [
            # Year + report type
            r'\d{4}\s*年\s*度\s*报\s*告',
            r'\d{4}\s*年\s*中\s*报',
            r'\d{4}\s*年\s*半\s*年\s*报',
            r'\d{4}\s*Annual\s*Report',
            r'\d{4}\s*Semi-annual\s*Report',

            # Dates (very common in headers)
            r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]',  # Full date: 2021-05-20 or 2021年5月20日
            r'\d{4}[-/]\d{1,2}[-/]',  # Date with trailing slash: 2021/5/
            r'\d{4}[-/]\d{1,2}$',  # Date without day: 2021/11 or 2021-05

            # Company info
            r'公\s*司\s*代\s*码\s*：\s*\d+',
            r'公\s*司\s*简\s*称\s*：\s*[\u4e00-\u9fff]+',
            r'代\s*码\s*：\s*\d{6}',
            r'Code\s*[:：]\s*\d+',  # for English

            # Stock codes
            r'\d{6}',  # 6-digit stock code
            r'[A-Z]{4}\s*\d{4}',  # Some stock formats like HK0700

            # Disclaimer/Statement text (often in headers)
            r'重\s*要\s*提\s*示',
            r'特\s*别\s*说\s*明',
            r'声\s*明',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _get_default_footer_patterns(self) -> list:
        """
        Get default regex patterns for identifying footer text.

        Returns:
            List of regex patterns
        """
        import re
        patterns = [
            # Page numbers
            r'-\s*\d+\s*-',
            r'第\s*\d+\s*页',
            r'Page\s*\d+',
            r'页\s*码\s*[:：]?\s*\d+',
            r'\d+\s*/\s*\d+',  # e.g., 1/215

            # Common footer text
            r'机\s*密',
            r'内\s*部\s*资\s*料',
            r'请\s*勿\s*外\s*传',
            r'协[\u3000 ]+会',  # Association
            r'SEMI',
            r'指\s*数',
            r'协\s*会',

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

            # Extract structured content
            self._extract_content(doc, result)

            logger.info(f"Successfully parsed {path.name}: "
                       f"{len(result.text_content)} text blocks, "
                       f"{len(result.tables)} tables, "
                       f"{len(result.images)} images, "
                       f"{len(result.structure)} structure elements")

            return result

        except Exception as e:
            logger.error(f"Error parsing PDF {path}: {str(e)}")
            raise

    def parse_batch(
        self,
        pdf_paths: List[Union[str, Path]],
        show_progress: bool = True
    ) -> List[ParseResult]:
        """
        Parse multiple PDF files.

        Args:
            pdf_paths: List of paths to PDF files
            show_progress: Whether to show progress

        Returns:
            List of ParseResult objects
        """
        results = []
        total = len(pdf_paths)

        for idx, pdf_path in enumerate(pdf_paths, 1):
            try:
                if show_progress:
                    logger.info(f"Progress: {idx}/{total}")

                result = self.parse_pdf(pdf_path)
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to parse {pdf_path}: {str(e)}")
                # Continue with next file

        logger.info(f"Batch processing complete: {len(results)}/{total} files parsed successfully")
        return results

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

            markdown_content = self._get_markdown_content(doc, add_page_markers)
            output = Path(output_path)
            self._save_markdown(output, markdown_content)

            logger.info(f"Saved Markdown to {output} ({len(markdown_content)} chars)")

            return markdown_content

        except Exception as e:
            logger.error(f"Error parsing PDF to Markdown {path}: {str(e)}")
            raise

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

        # Build a mapping of page numbers to page dimensions for header/footer filtering
        page_sizes_seen = {}
        if self.exclude_headers_footers:
            # doc.document.pages is a dict with page_no as key
            doc_pages = getattr(doc.document, 'pages', None)
            if doc_pages:
                for page_no, page_obj in doc_pages.items():
                    if hasattr(page_obj, 'size'):
                        page_sizes_seen[page_no] = page_obj.size
        repeat_texts = None
        if self.exclude_headers_footers and self.auto_header_footer and page_sizes_seen:
            repeat_texts = self._build_repeated_header_footer_texts(doc.document, page_sizes_seen)

        # Create a list of all elements with their page info and export function
        elements = []

        # Add text elements
        for text_item in doc.document.texts:
            page_no, bbox, page_idx = self._get_prov_info(text_item)

            # Skip headers and footers if enabled
            if self.exclude_headers_footers and bbox and page_idx in page_sizes_seen:
                bbox_list = list(bbox.as_tuple()) if hasattr(bbox, 'as_tuple') else list(bbox)
                text_content = text_item.text if hasattr(text_item, 'text') else str(text_item)
                if self._is_header_or_footer(bbox_list, page_sizes_seen[page_idx], text_content, repeat_texts):
                    continue

            elements.append({
                'type': 'text',
                'page_no': page_no,
                'bbox': bbox,
                'content': text_item.text if hasattr(text_item, 'text') else str(text_item)
            })

        # Add table elements
        for table_item in doc.document.tables:
            page_no, bbox, _ = self._get_prov_info(table_item)

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
            page_no, bbox, _ = self._get_prov_info(picture_item)

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
                # Note: in PDF coords, y increases upward, but we want reading order
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

    def save_batch_to_markdown(
        self,
        pdf_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        merge: bool = False,
        merged_filename: str = "merged_results.md",
        add_page_markers: bool = True
    ) -> None:
        """
        Parse batch PDFs and save as Markdown files.

        Args:
            pdf_paths: List of paths to PDF files
            output_dir: Directory to save Markdown files
            merge: Whether to merge all results into one file
            merged_filename: Filename for merged result
            add_page_markers: Whether to add page number markers
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if merge:
            # Merge all markdown into one file
            all_markdowns = []

            for idx, pdf_path in enumerate(pdf_paths, 1):
                try:
                    logger.info(f"Processing {idx}/{len(pdf_paths)}: {pdf_path}")
                    doc = self.converter.convert(str(pdf_path))

                    markdown_content = self._get_markdown_content(doc, add_page_markers)

                    # Add file name as header
                    pdf_name = Path(pdf_path).stem
                    section = f"\n\n# {pdf_name}\n\n{markdown_content}\n\n---\n"
                    all_markdowns.append(section)

                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {str(e)}")

            # Write merged file
            merged_file = output_path / merged_filename
            self._save_markdown(merged_file, "".join(all_markdowns))

            logger.info(f"Saved merged Markdown to {merged_file}")

        else:
            # Save each PDF as separate markdown file
            for pdf_path in pdf_paths:
                try:
                    doc = self.converter.convert(str(pdf_path))

                    markdown_content = self._get_markdown_content(doc, add_page_markers)

                    # Use filename without .pdf as markdown filename
                    md_name = Path(pdf_path).stem + ".md"
                    md_path = output_path / md_name

                    self._save_markdown(md_path, markdown_content)

                    logger.info(f"Saved {md_path}")

                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {str(e)}")

    def _extract_metadata(self, doc) -> dict:
        """Extract document metadata."""
        metadata = {}

        # Try to get metadata from the document
        if hasattr(doc, 'document') and hasattr(doc.document, 'meta'):
            meta = doc.document.meta
            for key, value in meta.items():
                metadata[key] = str(value) if value else None

        return metadata

    def _extract_content(self, doc, result: ParseResult) -> None:
        """
        Extract structured content from document.

        Args:
            doc: Docling document object
            result: ParseResult to populate
        """
        doc_content = doc.document

        # Extract all text blocks
        self._extract_all_text(doc_content, result)

        # Extract tables
        if self.extract_tables:
            self._extract_tables(doc_content, result)

        # Extract images
        if self.extract_images:
            self._extract_images(doc_content, result)

        # Extract document structure
        if self.extract_structure:
            self._extract_structure(doc_content, result)

    def _extract_all_text(self, doc_content, result: ParseResult) -> None:
        """Extract all text content with page information."""
        try:
            page_sizes_seen = self._get_page_sizes(doc_content) if self.exclude_headers_footers else {}
            repeat_texts = None
            if self.exclude_headers_footers and self.auto_header_footer and page_sizes_seen:
                repeat_texts = self._build_repeated_header_footer_texts(doc_content, page_sizes_seen)

            # Iterate through text elements
            for item in doc_content.texts:
                page_num, bbox_obj, page_idx = self._get_prov_info(item)
                bbox = list(bbox_obj.as_tuple()) if bbox_obj is not None else None

                # Try to get page size from doc_content.pages (which is a dict)
                if self.exclude_headers_footers and page_idx not in page_sizes_seen:
                    if hasattr(doc_content, 'pages') and page_idx in doc_content.pages:
                        page_obj = doc_content.pages[page_idx]
                        if hasattr(page_obj, 'size'):
                            page_sizes_seen[page_idx] = page_obj.size

                # Skip headers and footers if enabled
                if self.exclude_headers_footers and bbox and page_idx in page_sizes_seen:
                    if self._is_header_or_footer(bbox, page_sizes_seen[page_idx], item.text, repeat_texts):
                        continue

                text_content = TextContent(
                    text=item.text,
                    page_num=page_num,
                    bbox=bbox
                )
                result.text_content.append(text_content)

        except Exception as e:
            logger.warning(f"Error extracting text: {e}")

    def _is_header_or_footer(
        self,
        bbox: List[float],
        page_size,
        text: str,
        repeat_texts: Optional[Dict[str, set]] = None
    ) -> bool:
        """
        Check if text element is in header or footer area using position and pattern matching.

        Args:
            bbox: Bounding box as [left, bottom, right, top] (from bbox.as_tuple())
            page_size: Page size object with width and height attributes
            text: The text content to check for header/footer patterns
            repeat_texts: Optional repeated header/footer text sets

        Returns:
            True if the element should be excluded (is in header or footer)
        """
        in_header_area, in_footer_area = self._get_header_footer_area_flags(bbox, page_size)

        # If not in header or footer area, don't exclude
        if not (in_header_area or in_footer_area):
            return False

        # If pattern matching is disabled, exclude based on position only
        if not self.header_footer_patterns:
            return True

        # If in header area, check if text matches header patterns
        if in_header_area:
            if repeat_texts:
                text_norm = self._normalize_hf_text(text)
                if text_norm and text_norm in repeat_texts.get("header", set()):
                    return True
            return self._matches_header_pattern(text)

        # If in footer area, check if text matches footer patterns
        if in_footer_area:
            if repeat_texts:
                text_norm = self._normalize_hf_text(text)
                if text_norm and text_norm in repeat_texts.get("footer", set()):
                    return True
            return self._matches_footer_pattern(text)

        return False

    def _matches_header_pattern(self, text: str) -> bool:
        """
        Check if text matches header patterns.

        Args:
            text: The text to check

        Returns:
            True if text matches any header pattern
        """
        if not text:
            return False

        text_clean = text.strip()

        # Very short text (1-2 chars) is unlikely to be header
        if len(text_clean) <= 2:
            return False

        # Check against all header patterns
        for pattern in self.header_patterns:
            if pattern.search(text_clean):
                return True

        return False

    def _matches_footer_pattern(self, text: str) -> bool:
        """
        Check if text matches footer patterns.

        Args:
            text: The text to check

        Returns:
            True if text matches any footer pattern
        """
        if not text:
            return False

        text_clean = text.strip()

        # Very short text is unlikely to be footer
        if len(text_clean) <= 1:
            return False

        # Check against all footer patterns
        for pattern in self.footer_patterns:
            if pattern.search(text_clean):
                return True

        return False

    def _extract_tables(self, doc_content, result: ParseResult) -> None:
        """Extract all tables from the document."""
        try:
            for table in doc_content.tables:
                table_info = self._extract_table_info(table)
                if table_info:
                    result.tables.append(table_info)

        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")

    def _extract_table_info(self, table) -> Optional[TableInfo]:
        """Extract detailed information from a table."""
        try:
            # Get page number
            page_num, _, _ = self._get_prov_info(table)

            # Extract table data
            cells = []
            headers = []
            rows = 0
            cols = 0

            # Try to get grid data (structured table)
            if hasattr(table, 'grid'):
                grid = table.grid
                rows = len(grid) if grid else 0
                cols = len(grid[0]) if rows > 0 else 0

                # Extract cells
                for row_idx, row in enumerate(grid):
                    for col_idx, cell in enumerate(row):
                        # Extract cell text
                        text = ""
                        if isinstance(cell, DoclingTableCell):
                            text = cell.text
                        elif hasattr(cell, 'text'):
                            text = cell.text
                        else:
                            text = str(cell)

                        # First row as headers
                        if row_idx == 0:
                            headers.append(text)

                        cells.append(TableCell(
                            row=row_idx,
                            col=col_idx,
                            text=text
                        ))

            # Alternative: extract from texts attribute
            elif hasattr(table, 'texts'):
                for text_item in table.texts:
                    cells.append(TableCell(
                        row=0,
                        col=len(cells),
                        text=text_item.text if hasattr(text_item, 'text') else str(text_item)
                    ))

            return TableInfo(
                page_num=page_num,
                rows=rows,
                cols=cols,
                headers=headers,
                cells=cells
            )

        except Exception as e:
            logger.warning(f"Error extracting table info: {e}")
            return None

    def _extract_images(self, doc_content, result: ParseResult) -> None:
        """Extract all images from the document."""
        try:
            for picture in doc_content.pictures:
                image_info = self._extract_image_info(picture)
                if image_info:
                    result.images.append(image_info)

        except Exception as e:
            logger.warning(f"Error extracting images: {e}")

    def _extract_image_info(self, picture) -> Optional[ImageInfo]:
        """Extract detailed information from an image."""
        try:
            # Get page number
            page_num, _, _ = self._get_prov_info(picture)

            # Get bounding box
            bbox = [0.0, 0.0, 0.0, 0.0]
            _, bbox_obj, _ = self._get_prov_info(picture)
            if bbox_obj is not None and hasattr(bbox_obj, 'as_tuple'):
                bbox = list(bbox_obj.as_tuple())

            return ImageInfo(
                page_num=page_num,
                bbox=bbox,
                width=bbox[2] - bbox[0] if len(bbox) >= 4 else 0,
                height=bbox[3] - bbox[1] if len(bbox) >= 4 else 0
            )

        except Exception as e:
            logger.warning(f"Error extracting image info: {e}")
            return None

    def _extract_structure(self, doc_content, result: ParseResult) -> None:
        """Extract document structure (headings, titles, etc.)."""
        try:
            # Iterate through all items to find structure elements
            for item in doc_content.texts:
                structure = self._identify_structure(item)
                if structure:
                    result.structure.append(structure)

        except Exception as e:
            logger.warning(f"Error extracting structure: {e}")

    def _identify_structure(self, item) -> Optional[DocumentStructure]:
        """Identify if an item is a structural element and extract its info."""
        try:
            if not hasattr(item, 'label'):
                return None

            label = str(item.label).lower()

            # Check for structural elements
            structure_type = None
            level = None

            # Check for title
            if "title" in label:
                structure_type = "title"

            # Check for headers/headings
            elif "header" in label or "heading" in label:
                structure_type = "heading"
                # Extract heading level (1-6)
                for i in range(1, 7):
                    if str(i) in label or f"h{i}" in label or f"level-{i}" in label:
                        level = i
                        break

            # Only add if it's a structural element
            if structure_type:
                page_num, _, _ = self._get_prov_info(item)

                return DocumentStructure(
                    type=structure_type,
                    level=level,
                    text=item.text,
                    page_num=page_num
                )

        except Exception as e:
            logger.warning(f"Error identifying structure: {e}")

        return None
