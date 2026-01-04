#!/usr/bin/env python3
"""Crawl a URL (recursively) and save the content as a Markdown snapshot."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse

import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.recursive_url_loader import (
    _metadata_extractor,
    extract_sub_links,
)
from markdownify import markdownify


MAX_SNAPSHOT_BYTES = 40 * 1024 * 1024  # 40 MB cap for generated snapshot files


def html_to_markdown(html: str) -> str:
    """Convert raw HTML to Markdown while keeping headings/links readable."""
    return markdownify(html, heading_style="ATX")


@dataclass
class CrawlLogEntry:
    url: str
    depth: int
    status: str
    message: str
    title: str | None = None


def format_crawl_map(url: str, crawl_log: Iterable[CrawlLogEntry]) -> str:
    """Build Markdown-only crawl map."""
    entries = list(crawl_log)
    header = [
        f"# Crawl Map for {url}",
        "",
        f"_Captured {len(entries)} link{'s' if len(entries) != 1 else ''} during crawl._",
        "",
    ]

    lines: list[str] = []
    for entry in entries:
        indent = "  " * entry.depth
        title = entry.title or entry.url
        lines.append(
            f"{indent}- **{entry.status}** [{title}]({entry.url}) — {entry.message}"
        )
    lines.append("")
    return "\n".join(header + lines)


def format_markdown_sections(
    url: str, documents: Iterable[Document]
) -> tuple[str, list[str]]:
    """Build reusable Markdown sections (header + per-page blocks)."""
    docs = list(documents)
    header = [
        f"# Snapshot for {url}",
        "",
        f"_Crawled {len(docs)} page{'s' if len(docs) != 1 else ''}._",
        "",
    ]

    page_sections: list[str] = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source") or doc.metadata.get("url") or "Unknown source"
        title = doc.metadata.get("title")
        heading_text = title or source
        section_lines = [f"## {index}. {heading_text}"]
        if source and (not title or title != source):
            section_lines.append(f"[{source}]({source})")
        section_lines.append("")
        if doc.page_content:
            section_lines.append(doc.page_content.strip())
        else:
            section_lines.append("_No content extracted._")
        section_lines.append("")
        page_sections.append("\n".join(section_lines))

    header_text = "\n".join(header).rstrip() + "\n"
    return header_text, page_sections


def format_markdown(url: str, documents: Iterable[Document]) -> str:
    """Build a single Markdown string from the crawled documents."""
    header_text, page_sections = format_markdown_sections(url, documents)
    return header_text + "".join(page_sections)


def chunk_markdown_sections(
    header: str, pages: Sequence[str], *, max_bytes: int
) -> list[str]:
    """Split Markdown into file-sized parts without breaking pages."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    parts: list[str] = []
    current: list[str] = [header]
    current_size = len(header.encode("utf-8"))

    for index, page in enumerate(pages, start=1):
        page_size = len(page.encode("utf-8"))
        if page_size > max_bytes:
            raise ValueError(
                f"Page {index} is {page_size} bytes which exceeds the snapshot cap of {max_bytes} bytes. Pages are atomic and cannot be split."
            )

        if current_size + page_size > max_bytes:
            parts.append("".join(current))
            current = []
            current_size = 0

        current.append(page)
        current_size += page_size

    parts.append("".join(current))
    return parts


def default_output_path(url: str) -> Path:
    netloc = urlparse(url).netloc or "site"
    safe_netloc = netloc.replace(":", "_")
    return Path(f"{safe_netloc}.md")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("max-depth must be at least 1")
    return parsed


def crawl_site(
    url: str,
    *,
    max_depth: int,
    exclude_dirs: Sequence[str] | None,
    timeout: float,
    check_response_status: bool = True,
    continue_on_failure: bool = True,
) -> tuple[list[Document], list[CrawlLogEntry]]:
    """Recursively crawl a URL and capture both documents and a status log."""
    visited: set[str] = set()
    documents: list[Document] = []
    log_entries: list[CrawlLogEntry] = []
    base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/"
    excludes = tuple(exclude_dirs or ())

    def visit(current_url: str, depth: int) -> None:
        if depth >= max_depth:
            log_entries.append(
                CrawlLogEntry(
                    url=current_url,
                    depth=depth,
                    status="Warning",
                    message="Skipped (max depth reached)",
                )
            )
            return

        if current_url in visited:
            log_entries.append(
                CrawlLogEntry(
                    url=current_url,
                    depth=depth,
                    status="Debug",
                    message="Skipped (already visited)",
                )
            )
            return

        visited.add(current_url)
        try:
            response = requests.get(current_url, timeout=timeout)
            response.encoding = response.apparent_encoding
            if check_response_status and 400 <= response.status_code <= 599:
                raise ValueError(f"HTTP {response.status_code}")
        except Exception as exc:
            log_entries.append(
                CrawlLogEntry(
                    url=current_url,
                    depth=depth,
                    status="Error",
                    message=f"{exc.__class__.__name__}: {exc}",
                )
            )
            if not continue_on_failure:
                raise
            return

        raw_html = response.text
        metadata = _metadata_extractor(raw_html, current_url, response)
        title = metadata.get("title")
        content = html_to_markdown(raw_html)

        if content:
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )
            log_entries.append(
                CrawlLogEntry(
                    url=current_url,
                    depth=depth,
                    status="Info",
                    message=f"Fetched ({len(content)} chars)",
                    title=title,
                )
            )
        else:
            log_entries.append(
                CrawlLogEntry(
                    url=current_url,
                    depth=depth,
                    status="Warning",
                    message="Empty content extracted",
                    title=title,
                )
            )

        sub_links = extract_sub_links(
            raw_html,
            current_url,
            base_url=base_url,
            prevent_outside=True,
            exclude_prefixes=excludes,
            continue_on_failure=continue_on_failure,
        )

        for link in sub_links:
            visit(link, depth + 1)

    visit(url, 0)
    return documents, log_entries


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl a URL recursively and emit a Markdown file of the contents."
    )
    parser.add_argument("url", help="Root URL to crawl.")
    parser.add_argument(
        "-d",
        "--max-depth",
        type=positive_int,
        default=1,
        help="How deep to follow links (default: 1).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to write the Markdown (default: <domain>.md).",
    )
    parser.add_argument(
        "--exclude-dirs",
        nargs="*",
        default=None,
        help="Paths to skip when crawling (e.g. /blog /admin).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Request timeout in seconds for each page (default: 15.0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    documents, crawl_log = crawl_site(
        args.url,
        max_depth=args.max_depth,
        exclude_dirs=args.exclude_dirs,
        timeout=args.timeout,
    )

    if not documents:
        print(f"No documents found for {args.url}", file=sys.stderr)
        return 1

    output_path = args.output or default_output_path(args.url)
    crawl_map_path = output_path.with_name(
        f"{output_path.stem}-crawlmap{output_path.suffix}"
    )

    header_text, page_sections = format_markdown_sections(args.url, documents)
    markdown_parts = chunk_markdown_sections(
        header_text, page_sections, max_bytes=MAX_SNAPSHOT_BYTES
    )
    crawl_map_markdown = format_crawl_map(args.url, crawl_log)

    if len(markdown_parts) == 1:
        snapshot_paths = [output_path]
    else:
        snapshot_paths = [
            output_path.with_name(f"{output_path.stem}-{index}{output_path.suffix}")
            for index in range(1, len(markdown_parts) + 1)
        ]

    for path, content in zip(snapshot_paths, markdown_parts):
        path.write_text(content, encoding="utf-8")

    crawl_map_path.write_text(crawl_map_markdown, encoding="utf-8")

    if len(snapshot_paths) == 1:
        print(
            f"Wrote {len(documents)} page(s) to {snapshot_paths[0]} "
            f"(max {MAX_SNAPSHOT_BYTES // (1024 * 1024)} MB per file)"
        )
    else:
        print(
            f"Wrote {len(documents)} page(s) across {len(snapshot_paths)} files "
            f"(max {MAX_SNAPSHOT_BYTES // (1024 * 1024)} MB per file):"
        )
        for path in snapshot_paths:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  - {path} ({size_mb:.2f} MB)")

    print(f"Wrote crawl map to {crawl_map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
