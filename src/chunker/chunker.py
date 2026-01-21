"""
Markdown Document Chunker with Structure-Aware Splitting

This module chunks Markdown documents by:
1. Parsing the header hierarchy
2. Splitting by header boundaries first
3. Recursively splitting large chunks by paragraph/sentence
4. Protecting tables from being split
"""

import re
import uuid
from pathlib import Path
from typing import List, Tuple, Optional
import hashlib

from .models import Chunk, ChunkResult, HeaderNode
from .utils import logger, extract_page_numbers


class MarkdownChunker:
    """
    Chunk Markdown documents while preserving structure.

    Strategy:
    1. Parse header hierarchy (# ## ###)
    2. Split by header boundaries
    3. Detect and protect tables
    4. Recursively split large chunks
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        hard_limit: int = 4000,
        min_chunk_size: int = 100,
        preserve_tables: bool = True
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target character count per chunk
            chunk_overlap: Character overlap between chunks
            hard_limit: Maximum allowed characters per chunk
            min_chunk_size: Minimum characters for a valid chunk
            preserve_tables: Whether to keep tables intact
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.hard_limit = hard_limit
        self.min_chunk_size = min_chunk_size
        self.preserve_tables = preserve_tables

        # Compile regex patterns
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.table_pattern = re.compile(
            r'^[| :\-+\s]*\|[\s\|:+\-]*$|^\|?[\s\-:]+\|[\s\-:]+\|?',
            re.MULTILINE
        )
        self.page_marker_pattern = re.compile(r'\*{0,2}第\s*(\d+)\s*页\*{0,2}', re.IGNORECASE)

        logger.info(f"MarkdownChunker initialized (chunk_size={chunk_size}, "
                   f"overlap={chunk_overlap}, preserve_tables={preserve_tables})")

    def chunk(self, markdown_content: str, source_file: str = "") -> ChunkResult:
        """
        Chunk a Markdown document.

        Args:
            markdown_content: The Markdown text to chunk
            source_file: Source file path for metadata

        Returns:
            ChunkResult with all chunks
        """
        logger.info(f"Chunking document: {source_file}")

        # Step 1: Parse header tree
        header_tree = self._parse_header_tree(markdown_content)

        # Step 2: Extract sections based on headers
        sections = self._extract_sections(markdown_content, header_tree)

        # Step 3: Chunk each section
        all_chunks = []
        for section in sections:
            chunks = self._chunk_section(
                section['content'],
                section['title_path'],
                section['page_range'],
                source_file,
                section.get('page_number')
            )
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {source_file}")

        return ChunkResult(
            source_file=source_file,
            total_chunks=len(all_chunks),
            chunks=all_chunks,
            metadata={
                'original_length': len(markdown_content),
                'chunker_params': {
                    'chunk_size': self.chunk_size,
                    'overlap': self.chunk_overlap
                }
            }
        )

    def chunk_file(self, file_path: str) -> ChunkResult:
        """
        Chunk a Markdown file.

        Args:
            file_path: Path to the Markdown file

        Returns:
            ChunkResult with all chunks
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_text(encoding='utf-8')
        return self.chunk(content, str(path))

    def _parse_header_tree(self, content: str) -> List[HeaderNode]:
        """
        Parse Markdown content into a header tree.

        Args:
            content: Markdown text

        Returns:
            List of root-level HeaderNodes
        """
        headers = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            match = self.header_pattern.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headers.append({
                    'level': level,
                    'title': title,
                    'line_number': i,
                    'content_start': i + 1
                })

        # Build tree structure
        root_nodes = []
        stack = []  # (node, level)

        for header in headers:
            node = HeaderNode(
                level=header['level'],
                title=header['title'],
                line_number=header['line_number']
            )

            # Pop until we find the parent level
            while stack and stack[-1][1] >= node.level:
                stack.pop()

            if stack:
                # Add as child
                stack[-1][0].children.append(node)
                stack.append((node, node.level))
            else:
                # Root level
                root_nodes.append(node)
                stack.append((node, node.level))

        return root_nodes

    def _extract_sections(
        self,
        content: str,
        header_tree: List[HeaderNode]
    ) -> List[dict]:
        """
        Extract document sections based on header tree.

        Args:
            content: Full Markdown content
            header_tree: Parsed header tree

        Returns:
            List of sections with content and metadata
        """
        lines = content.split('\n')
        sections = []

        def extract_from_node(node: HeaderNode, title_path: List[str]):
            """Recursively extract sections from header nodes."""
            current_path = title_path + [node.title]

            # Find content range
            start_idx = node.line_number + 1

            # Find end index (next sibling or child)
            end_idx = len(lines)

            # Check if there's a next header at same or higher level
            siblings = self._get_next_siblings(lines, node.line_number, node.level)
            if siblings:
                end_idx = siblings[0]['line_number']

            # Extract content
            section_content = '\n'.join(lines[start_idx:end_idx]).strip()

            # Extract page range
            page_range = extract_page_numbers(section_content)

            # Add section
            if section_content:
                sections.append({
                    'content': section_content,
                    'title_path': current_path,
                    'page_range': page_range
                })

            # Recursively process children
            for child in node.children:
                extract_from_node(child, current_path)

        # Process all root nodes
        for node in header_tree:
            extract_from_node(node, [])

        # Handle content before first header
        if header_tree:
            first_header_line = header_tree[0].line_number
            if first_header_line > 0:
                preamble = '\n'.join(lines[:first_header_line]).strip()
                if preamble:
                    # Split by page markers first
                    page_sections = self._split_by_page_markers(preamble)
                    for sec in page_sections:
                        page_range = extract_page_numbers(sec['content'])
                        sec['page_range'] = page_range
                        sections.insert(0, sec)
        else:
            # No headers found, split by page markers
            full_content = '\n'.join(lines).strip()
            page_sections = self._split_by_page_markers(full_content)
            for sec in page_sections:
                page_range = extract_page_numbers(sec['content'])
                sec['page_range'] = page_range
                sections.append(sec)

        return sections

    def _split_by_page_markers(self, content: str) -> List[dict]:
        """
        Split content by page markers (**第 X 页**).

        Args:
            content: Content to split

        Returns:
            List of sections with content
        """
        sections = []
        lines = content.split('\n')
        current_section_lines = []
        current_page = None

        page_marker_pattern = re.compile(r'^\*{0,2}第\s*(\d+)\s*页\*{0,2}$', re.IGNORECASE)

        for line in lines:
            match = page_marker_pattern.match(line.strip())
            if match:
                # Save previous section
                if current_section_lines:
                    section_content = '\n'.join(current_section_lines).strip()
                    if section_content:
                        sections.append({
                            'content': section_content,
                            'title_path': [],
                            'page_number': current_page
                        })

                # Start new section
                current_page = int(match.group(1))
                current_section_lines = []
            else:
                current_section_lines.append(line)

        # Don't forget the last section
        if current_section_lines:
            section_content = '\n'.join(current_section_lines).strip()
            if section_content:
                sections.append({
                    'content': section_content,
                    'title_path': [],
                    'page_number': current_page
                })

        return sections

    def _get_next_siblings(
        self,
        lines: List[str],
        current_line: int,
        level: int
    ) -> List[dict]:
        """Find next headers at same or higher level."""
        siblings = []
        for i, line in enumerate(lines):
            if i <= current_line:
                continue
            match = self.header_pattern.match(line)
            if match:
                header_level = len(match.group(1))
                if header_level <= level:
                    siblings.append({
                        'level': header_level,
                        'title': match.group(2).strip(),
                        'line_number': i
                    })
        return siblings

    def _chunk_section(
        self,
        content: str,
        title_path: List[str],
        page_range: Optional[List[int]] = None,
        source_file: str = "",
        page_number: Optional[int] = None
    ) -> List[Chunk]:
        """
        Chunk a section of content.

        Args:
            content: Section content
            title_path: Header hierarchy
            page_range: Page number range
            source_file: Source file path for metadata
            page_number: Single page number (if available)

        Returns:
            List of chunks
        """
        # Use page_number if available, otherwise use page_range
        if page_number is not None:
            page_range = [page_number, page_number]

        # Check if content is small enough
        if len(content) <= self.chunk_size:
            return [self._create_chunk(content, title_path, page_range, source_file)]

        # Detect and protect tables
        if self.preserve_tables:
            table_ranges = self._detect_table_ranges(content)
            if table_ranges:
                return self._chunk_with_tables(content, title_path, page_range, table_ranges, source_file)

        # Recursive splitting
        return self._recursive_split(content, title_path, page_range, source_file)

    def _detect_table_ranges(self, content: str) -> List[Tuple[int, int]]:
        """
        Detect table ranges in content.

        Returns:
            List of (start, end) character indices
        """
        ranges = []
        lines = content.split('\n')
        in_table = False
        table_start = 0

        for i, line in enumerate(lines):
            is_table_line = bool(self.table_pattern.match(line)) or '|' in line

            if is_table_line and not in_table:
                # Start of table
                in_table = True
                table_start = i
            elif not is_table_line and in_table:
                # End of table
                in_table = False
                table_end = i
                ranges.append((table_start, table_end))

        # Handle table at end
        if in_table:
            ranges.append((table_start, len(lines)))

        return ranges

    def _chunk_with_tables(
        self,
        content: str,
        title_path: List[str],
        page_range: Optional[List[int]],
        table_ranges: List[Tuple[int, int]],
        source_file: str
    ) -> List[Chunk]:
        """
        Chunk content while preserving table integrity.
        """
        chunks = []
        lines = content.split('\n')
        current_section = []
        current_size = 0

        table_ranges_set = set((start, end) for start, end in table_ranges)

        for i, line in enumerate(lines):
            # Check if this line is part of a table
            in_table = any(start <= i < end for start, end in table_ranges)

            if in_table:
                # Flush current section before table
                if current_section:
                    section_text = '\n'.join(current_section)
                    if len(section_text) >= self.min_chunk_size:
                        chunks.extend(self._recursive_split(section_text, title_path, page_range, source_file))
                    current_section = []
                    current_size = 0

                # Add table as its own chunk
                # Find the end of this table
                table_end = next(end for start, end in table_ranges if start <= i < end)
                table_lines = lines[i:table_end]
                table_text = '\n'.join(table_lines)

                chunks.append(self._create_chunk(table_text, title_path, page_range, source_file))
                i = table_end - 1  # Skip to end of table
            else:
                current_section.append(line)
                current_size += len(line) + 1

                # Check if we need to split
                if current_size >= self.chunk_size:
                    section_text = '\n'.join(current_section)
                    chunks.extend(self._recursive_split(section_text, title_path, page_range, source_file))
                    current_section = []
                    current_size = 0

        # Handle remaining content
        if current_section:
            section_text = '\n'.join(current_section)
            if len(section_text) >= self.min_chunk_size:
                chunks.extend(self._recursive_split(section_text, title_path, page_range, source_file))

        return chunks

    def _recursive_split(
        self,
        content: str,
        title_path: List[str],
        page_range: Optional[List[int]],
        source_file: str
    ) -> List[Chunk]:
        """
        Recursively split content by paragraph -> sentence -> fixed window.
        """
        if len(content) <= self.chunk_size:
            return [self._create_chunk(content, title_path, page_range, source_file)]

        # Try splitting by double newline (paragraphs)
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            return self._split_by_delimiter(paragraphs, '\n\n', title_path, page_range, source_file)

        # Try splitting by single newline
        lines = content.split('\n')
        if len(lines) > 1:
            return self._split_by_delimiter(lines, '\n', title_path, page_range, source_file)

        # Try splitting by sentence
        sentences = re.split(r'([。！？.!?])', content)
        if len(sentences) > 1:
            # Rejoin punctuation with sentences
            sentences = [''.join(pair) for pair in zip(sentences[::2], sentences[1::2])]
            if sentences[-1] == '':
                sentences.pop()
            return self._split_by_delimiter(sentences, '', title_path, page_range, source_file)

        # Fixed window split as last resort
        return self._fixed_window_split(content, title_path, page_range, source_file)

    def _split_by_delimiter(
        self,
        pieces: List[str],
        delimiter: str,
        title_path: List[str],
        page_range: Optional[List[int]],
        source_file: str
    ) -> List[Chunk]:
        """
        Split content by a delimiter while respecting chunk size.
        """
        chunks = []
        current_chunk = ""
        delimiter_len = len(delimiter)

        for piece in pieces:
            test_chunk = current_chunk + (delimiter if current_chunk else '') + piece

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it's large enough
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(current_chunk, title_path, page_range, source_file))

                # Handle pieces that are too large
                if len(piece) > self.hard_limit:
                    # Recursively split large piece
                    sub_chunks = self._fixed_window_split(piece, title_path, page_range, source_file)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                elif len(piece) >= self.min_chunk_size:
                    # Piece is a valid chunk on its own
                    chunks.append(self._create_chunk(piece, title_path, page_range, source_file))
                    current_chunk = ""
                else:
                    # Piece too small, add to next chunk with overlap
                    if current_chunk:
                        # Add overlap from previous chunk
                        overlap = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                        current_chunk = overlap + delimiter + piece
                    else:
                        current_chunk = piece

        # Handle remaining content
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._create_chunk(current_chunk, title_path, page_range, source_file))

        return chunks

    def _fixed_window_split(
        self,
        content: str,
        title_path: List[str],
        page_range: Optional[List[int]],
        source_file: str
    ) -> List[Chunk]:
        """
        Split content using fixed sliding window.
        """
        chunks = []
        start = 0
        content_len = len(content)

        while start < content_len:
            end = min(start + self.chunk_size, content_len)
            chunk_content = content[start:end]

            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(self._create_chunk(chunk_content, title_path, page_range, source_file))

            start = end - self.chunk_overlap

        return chunks

    def _create_chunk(
        self,
        content: str,
        title_path: List[str],
        page_range: Optional[List[int]],
        source_file: str = ""
    ) -> Chunk:
        """Create a Chunk object with metadata."""
        # Generate unique ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        chunk_id = f"chunk_{content_hash}_{uuid.uuid4().hex[:8]}"

        # Format page for metadata
        if page_range:
            if page_range[0] == page_range[1]:
                page = page_range[0]
            else:
                page = f"{page_range[0]}-{page_range[1]}"
        else:
            page = None

        # Format section path
        section = ' > '.join(title_path) if title_path else None

        return Chunk(
            chunk_id=chunk_id,
            content=content.strip(),
            title_path=title_path,
            page_range=page_range,
            char_count=len(content.strip()),
            metadata={
                'source': source_file,
                'page': page,
                'section': section
            }
        )
