"""
Utility functions for PDF parsing
"""
# 日志配置 (Logging):

# 它配置了 Python 的标准日志系统，定义了输出格式（时间 - 模块名 - 等级 - 消息），以便程序能清晰地打印出运行状态和错误信息。
import logging
from pathlib import Path
from typing import Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_pdf_path(pdf_path: Union[str, Path]) -> Path:
    """
    Validate that the given path is a valid PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Path object if valid

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a PDF
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    if path.suffix.lower() != '.pdf':
        raise ValueError(f"File is not a PDF: {pdf_path}")

    return path.resolve()


def get_pdf_files(directory: Union[str, Path]) -> list[Path]:
    """
    Get all PDF files in a directory.

    Args:
        directory: Directory path to search

    Returns:
        List of Path objects for PDF files
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    pdf_files = sorted(dir_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

    return pdf_files
