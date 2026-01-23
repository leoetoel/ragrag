import os
import re
import json
from typing import List, Dict, Tuple, Optional
from loguru import logger


class AnnualReportChunker:
    """年报结构化分块器"""

    def __init__(self):
        # 年报常见章节标题模式（按优先级排序）
        self.section_patterns = [
            # 第X节 格式（最明确）
            r'^第[一二三四五六七八九十百]+节\s+\S.*$',
            r'^第\d+\s*节\s+\S.*$',

            # Markdown标题格式
            r'^#+\s+\S.*$',  # # 标题

            # 一级标题：中文数字 + 、
            r'^[一二三四五六七八九十百]+[、．.]\s+\S.*$',  # 确保标题后有内容

            # 目录/重要提示/释义（单独成行）
            r'^重要提示\s*$',
            r'^目录\s*$',
            r'^释义\s*$',

            # 常见章节名称（完整匹配）
            r'^公司简介\s*$',
            r'^会计数据\s*$',
            r'^财务报告\s*$',
            r'^董事会报告\s*$',
            r'^监事会报告\s*$',
            r'^重要事项\s*$',
            r'^股本变动\s*$',
            r'^股东信息\s*$',
            r'^公司债券\s*$',
            r'^财务报表\s*$',
        ]

        # 子章节标题模式（用于识别小节）
        self.subsection_patterns = [
            r'^#+\s+\S.*$',  # Markdown标题
            r'^[（(][一二三四五六七八九十]+[)）]\s*.+',
            r'^[（(]\d+[)）]\s*.+',
            r'^[①②③④⑤⑥⑦⑧⑨⑩]\s*.+',
        ]

        # 表格相关关键词
        self.table_header_keywords = [
            '项目', '科目', '名称', '说明', '附注', '单位', '币种',
            '金额', '数量', '比例', '比率', '%', '元', '万元', '亿元'
        ]

        self.finance_keywords = [
            '资产', '负债', '权益', '所有者权益', '资产负债表',
            '利润', '损益', '收入', '成本', '费用', '利润表', '损益表',
            '现金流', '现金流量表', '股东', '股本', '股份'
        ]

    def is_section_title(self, line: str) -> Tuple[bool, str]:
        """
        判断一行是否是章节标题
        返回: (是否是标题, 标题级别: 'main'/'sub'/'none')
        """
        line = line.strip()

        # 检查是否是一级章节标题
        for pattern in self.section_patterns:
            if re.match(pattern, line):
                return True, 'main'

        # 检查是否是子章节标题
        for pattern in self.subsection_patterns:
            if re.match(pattern, line):
                return True, 'sub'

        return False, 'none'

    def clean_line(self, line: str) -> str:
        """清理行内容"""
        line = line.strip()
        # 移除多余空格
        line = re.sub(r'\s+', ' ', line)
        return line

    def is_markdown_metadata(self, line: str) -> bool:
        """判断是否是Markdown元数据"""
        line = line.strip()
        return line.startswith('---') or line.startswith('```') or line.startswith('|--')

    def extract_sections_from_text(self, text: str) -> List[Dict]:
        """
        从纯文本中提取章节
        返回: [{'title': 章节标题, 'level': 级别, 'content': 内容, 'lines': 行号列表}]
        """
        lines = text.split('\n')
        sections = []
        current_section = None
        line_number = 0

        for line in lines:
            line_number += 1
            line = self.clean_line(line)

            # 跳过空行和Markdown元数据
            if not line or self.is_markdown_metadata(line):
                continue

            # 检查是否是章节标题
            is_title, level = self.is_section_title(line)

            if is_title:
                # 保存上一个章节
                if current_section:
                    sections.append(current_section)

                # 创建新章节
                current_section = {
                    'title': line,
                    'level': level,
                    'content': '',
                    'lines': [line_number],
                    'char_start': len(text[:text.index(line)]) if line in text else 0
                }
            else:
                # 添加到当前章节
                if current_section:
                    current_section['content'] += line + '\n'
                    current_section['lines'].append(line_number)
                else:
                    # 文档开头的内容（在第一个章节之前的）
                    current_section = {
                        'title': '文档开头',
                        'level': 'main',
                        'content': line + '\n',
                        'lines': [line_number],
                        'char_start': 0
                    }

        # 保存最后一个章节
        if current_section:
            sections.append(current_section)

        return sections

    def merge_small_sections(self, sections: List[Dict], min_chars: int = 200) -> List[Dict]:
        """
        改进的章节合并：不合并包含表格的章节
        min_chars: 最小字符数，小于此值的章节会被合并
        """
        if not sections:
            return sections

        merged = []
        i = 0

        while i < len(sections):
            current = sections[i]

            # 检查当前章节是否包含表格
            current_has_table = '|' in current['content']

            # 如果当前章节太小且不是第一个
            if len(current['content']) < min_chars and merged and not current_has_table:
                last_section = merged[-1]
                last_has_table = '|' in last_section['content']

                # 如果上一个章节有表格，不合并（保持表格独立）
                if last_has_table:
                    merged.append(current)
                else:
                    # 合并逻辑
                    last_section['content'] += '\n\n' + current['content']
                    # 标题处理：使用更有意义的合并
                    if '+' not in last_section['title']:
                        last_section['title'] = f"{last_section['title']} [+ {current['title']}]"
                    else:
                        # 如果已经有合并标记，简化显示
                        last_section['title'] = last_section['title'].split(' [')[0] + ' [...]'
                    last_section['lines'].extend(current['lines'])
            else:
                merged.append(current)

            i += 1

        return merged

    def chunk_by_sections(
            self,
            text: str,
            min_chars: int = 100,
            max_chars: int = 3000,
            merge_small: bool = True
    ) -> List[Dict]:
        """
        按章节进行分块

        参数:
            text: 输入文本
            min_chars: 最小字符数（用于合并小章节）
            max_chars: 最大字符数（超过此大小的章节会进一步分割）
            merge_small: 是否合并小章节

        返回:
            章节块列表
        """
        # 提取章节
        sections = self.extract_sections_from_text(text)

        if merge_small:
            sections = self.merge_small_sections(sections, min_chars)

        # 对过大的章节进行分割
        final_chunks = []
        for section in sections:
            if len(section['content']) <= max_chars:
                final_chunks.append(section)
            else:
                # 分割大章节
                sub_chunks = self.split_large_section(section, max_chars)
                final_chunks.extend(sub_chunks)

        return final_chunks

    def split_large_section(self, section: Dict, max_chars: int) -> List[Dict]:
        """
        将过大的章节分割成多个块
        策略：
        1. 先按子章节分割（如果有）
        2. 再按段落分割，保护表格完整性
        """
        content = section['content']
        title = section['title']
        level = section['level']

        # 检查是否有Markdown标题作为子章节
        lines = content.split('\n')
        has_subheadings = any(re.match(r'^#+\s+', line.strip()) for line in lines)

        if has_subheadings:
            return self._split_by_markdown_headings(section, max_chars)

        # 没有子章节，按智能段落分割（包含表格保护）
        return self._split_by_smart_paragraphs(title, content, level, max_chars)

    def _split_by_markdown_headings(self, section: Dict, max_chars: int) -> List[Dict]:
        """按Markdown标题分割大章节"""
        content = section['content']
        title = section['title']
        level = section['level']

        chunks = []
        current_content = ''
        current_heading = title
        chunk_num = 1

        lines = content.split('\n')
        for line in lines:
            # 检查是否是Markdown标题
            if re.match(r'^#+\s+', line.strip()):
                # 保存当前块
                if current_content.strip():
                    chunks.append({
                        'title': current_heading,
                        'content': current_content.strip(),
                        'level': level
                    })
                    chunk_num += 1

                # 开始新块
                current_heading = f"{title} - {line.strip()}"
                current_content = line + '\n'
            else:
                current_content += line + '\n'

        # 保存最后一块
        if current_content.strip():
            chunks.append({
                'title': current_heading,
                'content': current_content.strip(),
                'level': level
            })

        return chunks

    def _split_by_smart_paragraphs(self, title: str, content: str, level: str, max_chars: int) -> List[Dict]:
        """
        智能按段落分割，完整保护表格结构 (已修复死循环Bug)
        """
        lines = content.split('\n')
        chunks = []
        current_chunk_lines = []
        current_size = 0
        chunk_num = 1

        i = 0
        while i < len(lines):
            line = lines[i]
            line_size = len(line)

            # 🔴【检测表格开始】
            if self._is_table_start(line, i, lines):
                # 找到表格结束位置
                table_start = i
                table_end = self._find_table_end(i, lines)

                # 🔥【关键修复】：防止死循环
                # 如果检测逻辑矛盾，find_table_end 返回了原地，或者没有前进
                if table_end <= table_start:
                    # 此时被误判为表格头，但实际上无法提取表格
                    # 当作普通文本处理，强制跳过当前行
                    current_chunk_lines.append(line)
                    current_size += line_size
                    i += 1
                    continue

                # 提取完整表格
                table_lines = lines[table_start:table_end]
                table_content = '\n'.join(table_lines)

                # 1. 先保存当前已积累的文本块
                if current_chunk_lines:
                    chunks.append({
                        'title': f"{title} ({chunk_num})" if chunk_num > 1 else title,
                        'content': '\n'.join(current_chunk_lines),
                        'level': level,
                        'has_table': False
                    })
                    chunk_num += 1
                    current_chunk_lines = []
                    current_size = 0

                # 2. 保存表格块
                table_title = f"{title} - 表格({chunk_num})"
                if chunk_num == 1:
                    table_title = f"{title} - 表格"

                chunks.append({
                    'title': table_title,
                    'content': table_content,
                    'level': level,
                    'has_table': True,
                    'table_type': self._detect_table_type(table_lines)
                })
                chunk_num += 1

                # 3. 移动索引到表格结束处
                i = table_end
                continue

            # --- 普通段落处理逻辑 ---
            if current_size + line_size > max_chars and current_chunk_lines:
                chunks.append({
                    'title': f"{title} ({chunk_num})" if chunk_num > 1 else title,
                    'content': '\n'.join(current_chunk_lines),
                    'level': level,
                    'has_table': False
                })
                chunk_num += 1
                current_chunk_lines = []
                current_size = 0

            current_chunk_lines.append(line)
            current_size += line_size
            i += 1

        # 保存最后一块
        if current_chunk_lines:
            chunks.append({
                'title': f"{title} ({chunk_num})" if chunk_num > 1 else title,
                'content': '\n'.join(current_chunk_lines),
                'level': level,
                'has_table': False
            })

        return chunks

    def _is_table_start(self, line: str, index: int, lines: List[str]) -> bool:
        """
        检测是否是表格开始
        Markdown表格特征：
        1. 以 | 开头或包含 |
        2. 下一行是分隔线（包含 - 和 |）
        """
        line = line.strip()

        # 1. 必须有|符号
        if '|' not in line:
            return False

        # 2. 排除某些特殊情况
        # - 排除章节标题
        if self.is_section_title(line)[0]:
            return False

        # - 排除列表项
        if re.match(r'^[*-]\s+', line):
            return False

        # 3. 检查表格特征
        pipe_count = line.count('|')

        # 3.1 检查是否是简单的表格行（列数合理）
        if pipe_count < 2 or pipe_count > 30:  # 2-30列之间
            return False

        # 3.2 检查是否包含表格内容（数字、中文、空格）
        # 移除|符号和空格，检查剩余内容
        content = line.replace('|', '').replace(' ', '')
        if not content:
            return False

        # 3.3 检查下一行是否是分隔线
        if index + 1 < len(lines):
            next_line = lines[index + 1].strip()
            if '|' in next_line:
                # 计算分隔线特征：包含多个连续的-或=
                sep_pattern = r'[-=]+'
                sep_parts = re.split(r'\|', next_line)
                if len(sep_parts) > 1:
                    has_separator = any(re.match(sep_pattern, part.strip()) for part in sep_parts if part.strip())
                    if has_separator:
                        return True

        # 3.4 检查是否是财报表格（包含关键词）
        finance_keywords = self.finance_keywords + self.table_header_keywords
        if any(keyword in line for keyword in finance_keywords):
            # 确认有足够的列
            if pipe_count >= 3:
                return True

        # 3.5 检查是否是连续表格行
        if index > 0:
            prev_line = lines[index - 1].strip()
            if '|' in prev_line and not self.is_section_title(prev_line)[0]:
                # 上一行也是表格行，且不是标题
                # 检查是否有合理的表格内容
                prev_parts = [p.strip() for p in prev_line.split('|') if p.strip()]
                curr_parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(prev_parts) >= 2 and len(curr_parts) >= 2:
                    return True

        return False

    def _find_table_end(self, start_index: int, lines: List[str]) -> int:
        """
        精确找到表格结束位置
        """
        i = start_index
        consecutive_table_rows = 0

        while i < len(lines):
            line = lines[i].strip()

            # 空行且已经有表格内容，则结束
            if not line and consecutive_table_rows > 0:
                return i

            # 检查是否是表格行
            is_table_row = self._is_table_row(line, i, lines)

            if is_table_row:
                consecutive_table_rows += 1
                i += 1
                continue
            else:
                # 不是表格行
                if consecutive_table_rows > 0:
                    # 已经有表格内容，当前行不是表格，结束表格
                    return i
                else:
                    # 根本没有表格，返回原位置
                    return start_index

            i += 1

        return len(lines)

    def _is_table_row(self, line: str, index: int, lines: List[str]) -> bool:
        """
        判断是否是表格行（比_start更宽松，用于检测连续表格行）
        """
        if not line or '|' not in line:
            return False

        # 排除明显不是表格的情况
        if line.startswith('#') or self.is_section_title(line)[0]:
            return False

        # 检查是否有合理的内容
        parts = [p.strip() for p in line.split('|') if p.strip()]

        # 空单元格太多的情况排除
        if len(parts) < 2:
            return False

        # 检查内容特征
        # 表格内容通常包含：数字、中文、少量特殊字符
        valid_content = False
        for part in parts:
            if part:
                # 包含数字或中文
                if re.search(r'[\d一二三四五六七八九十百千万亿%.,]', part) or re.search(r'[\u4e00-\u9fff]', part):
                    valid_content = True
                    break

        return valid_content

    def _detect_table_type(self, table_lines: List[str]) -> str:
        """
        检测表格类型
        """
        if not table_lines:
            return 'unknown'

        # 检查常见的财务报表表头
        # 查看前3行，因为可能有复杂的多行表头
        header_text = ' '.join(table_lines[:min(3, len(table_lines))]).lower()

        if any(keyword in header_text for keyword in ['资产', '负债', '所有者权益', '资产负债表']):
            return 'balance_sheet'
        elif any(keyword in header_text for keyword in ['利润', '损益', '收入', '费用', '利润表', '损益表']):
            return 'income_statement'
        elif any(keyword in header_text for keyword in ['现金流', '现金', '现金流量']):
            return 'cash_flow'
        elif any(keyword in header_text for keyword in ['股东', '股本', '股份', '所有者权益变动']):
            return 'equity'
        elif any(keyword in header_text for keyword in ['审计', '会计师', '审计报告']):
            return 'audit'
        else:
            return 'general'

    def chunk_by_sections_with_sliding_window(
            self,
            text: str,
            section_max_chars: int = 2000,
            sliding_window_size: int = 1000,
            sliding_overlap: int = 200,
            merge_small: bool = True
    ) -> List[Dict]:
        """
        混合分块策略：先结构化分块，大章节使用滑窗

        参数:
            text: 输入文本
            section_max_chars: 章节最大字符数，超过则使用滑窗
            sliding_window_size: 滑窗大小
            sliding_overlap: 滑窗重叠大小
            merge_small: 是否合并小章节

        返回:
            分块列表
        """
        # 1. 先进行结构化分块
        sections = self.extract_sections_from_text(text)

        if merge_small:
            sections = self.merge_small_sections(sections, min_chars=100)

        # 2. 对每个章节判断是否需要滑窗
        final_chunks = []
        for section in sections:
            content_len = len(section['content'])

            if content_len <= section_max_chars:
                # 小章节，直接保留
                final_chunks.append(section)
            else:
                # 大章节，使用滑窗分块
                logger.info(f'章节 "{section["title"][:30]}..." 大小 {content_len} 字符，使用滑窗分块')

                sliding_chunks = self._sliding_window_by_char(
                    title=section['title'],
                    content=section['content'],
                    level=section['level'],
                    chunk_size=sliding_window_size,
                    overlap=sliding_overlap
                )

                final_chunks.extend(sliding_chunks)

        return final_chunks

    def _sliding_window_by_char(
            self,
            title: str,
            content: str,
            level: str,
            chunk_size: int,
            overlap: int
    ) -> List[Dict]:
        """
        按字符滑窗分块
        优先在句子/段落边界切分
        """
        chunks = []
        start = 0
        content_len = len(content)
        chunk_num = 1

        while start < content_len:
            # 计算窗口结束位置
            end = min(start + chunk_size, content_len)

            # 如果不是最后一块，尝试在句子边界切分
            if end < content_len:
                # 优先找段落边界（\n\n）
                paragraph_boundary = content.rfind('\n\n', start, end)
                if paragraph_boundary > start + chunk_size * 0.7:  # 至少保留70%
                    end = paragraph_boundary + 2
                else:
                    # 其次找句子边界（句号）
                    sentence_boundary = content.rfind('。', start, end)
                    if sentence_boundary > start + chunk_size * 0.7:
                        end = sentence_boundary + 1
                    else:
                        # 最后找换行
                        line_boundary = content.rfind('\n', start, end)
                        if line_boundary > start + chunk_size * 0.7:
                            end = line_boundary + 1

            # 提取窗口内容
            chunk_content = content[start:end].strip()

            if chunk_content:
                chunks.append({
                    'title': f"{title} (滑动{chunk_num})" if chunk_num > 1 else title,
                    'content': chunk_content,
                    'level': level,
                    'char_range': [start, end],
                    'overlap': overlap if chunk_num > 1 else 0
                })
                chunk_num += 1

            # 移动窗口（保留重叠）
            start = end - overlap if end < content_len else content_len

        return chunks


def load_md_file(file_path: str) -> List[str]:
    """
    加载Markdown文件

    参数:
        file_path: Markdown文件路径

    返回:
        内容列表，每个元素为一页（这里将整个文件作为一页处理）
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        logger.info(f"成功加载Markdown文件: {file_path} (大小: {len(content)} 字符)")
        return [content]  # 返回列表格式以保持接口一致性

    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        return []
    except Exception as e:
        logger.error(f"读取Markdown文件失败: {e}")
        return []


def find_md_files(data_path: str) -> List[str]:
    """
    在指定目录下查找所有Markdown文件

    参数:
        data_path: 数据目录路径

    返回:
        Markdown文件路径列表
    """
    md_files = []

    # 检查是否为文件
    if os.path.isfile(data_path) and data_path.endswith('.md'):
        return [data_path]

    # 检查是否为目录
    if os.path.isdir(data_path):
        for file_name in os.listdir(data_path):
            if file_name.endswith('.md'):
                file_path = os.path.join(data_path, file_name)
                md_files.append(file_path)

    logger.info(f"在 {data_path} 中找到 {len(md_files)} 个Markdown文件")
    return md_files


def chunk_md_by_sections(file_path: str) -> List[Dict]:
    """
    对Markdown文件按章节进行分块

    参数:
        file_path: Markdown文件路径

    返回:
        章节块列表
    """
    chunker = AnnualReportChunker()
    pages = load_md_file(file_path)

    if not pages:
        logger.warning(f'未加载到页面内容: {file_path}')
        return []

    # 合并所有页面文本
    full_text = '\n\n'.join(pages)

    # 按章节分块
    chunks = chunker.chunk_by_sections(
        full_text,
        min_chars=100,
        max_chars=3000,
        merge_small=True
    )

    return chunks


def clean_text(text):
    """清理文本"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_text_chunks_from_md(file_path: str, save_json: bool = True) -> List[str]:
    """
    从Markdown文件获取文本分块

    参数:
        file_path: Markdown文件路径
        save_json: 是否保存为JSON文件

    返回:
        格式化后的文本块列表
    """
    pages = load_md_file(file_path)
    if not pages:
        return []

    full_text = "\n\n".join(pages)
    chunker = AnnualReportChunker()

    # 执行结构化分块
    structured_chunks = chunker.chunk_by_sections(
        full_text,
        min_chars=200,
        max_chars=800,
        merge_small=True
    )

    # --- 保存 JSON 文件 ---
    if save_json:
        # 从文件路径生成JSON文件名
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        json_filename = f"{base_name}_chunks.json"

        # 保存到同目录
        output_dir = os.path.dirname(file_path) or '.'
        output_path = os.path.join(output_dir, json_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_chunks, f, ensure_ascii=False, indent=4)
        logger.info(f"分块 JSON 已保存至: {output_path}")

    # 格式转换供检索使用
    final_text_list = []
    for item in structured_chunks:
        title = item.get('title', '未知章节')
        content = item.get('content', '').strip()
        has_table = item.get('has_table', False)
        table_type = item.get('table_type', '')

        if has_table:
            formatted_text = f"【表格：{table_type}】\n{content}"
        else:
            formatted_text = f"【章节：{title}】\n{content}"
        final_text_list.append(formatted_text)

    logger.info(f"结构化分块完成，共生成 {len(final_text_list)} 个切片")
    return final_text_list
