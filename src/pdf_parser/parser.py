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
        custom_footer_patterns: list = None
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
