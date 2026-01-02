#!/usr/bin/env python3
"""Crawl a URL (recursively) and save the content as a Markdown snapshot."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse

from langchain_community.document_loaders import RecursiveUrlLoader
from markdownify import markdownify


def html_to_markdown(html: str) -> str:
    """Convert raw HTML to Markdown while keeping headings/links readable."""
    return markdownify(html, heading_style="ATX")


def build_loader(
    url: str, max_depth: int, exclude_dirs: Sequence[str] | None, timeout: float
) -> RecursiveUrlLoader:
    """Create a RecursiveUrlLoader while staying compatible with loader signature."""
    loader_kwargs = {
        "url": url,
        "max_depth": max_depth,
        "extractor": html_to_markdown,
    }

    optional_args = {
        "continue_on_failure": True,
        "check_response_status": True,
        "exclude_dirs": exclude_dirs,
        "timeout": timeout,
    }

    supported_params = set(inspect.signature(RecursiveUrlLoader).parameters)
    for name, value in optional_args.items():
        if name in supported_params and value is not None:
            loader_kwargs[name] = value

    return RecursiveUrlLoader(**loader_kwargs)  # type: ignore[arg-type]


def format_markdown(url: str, documents: Iterable) -> str:
    """Build a single Markdown string from the crawled documents."""
    docs = list(documents)
    header = [
        f"# Snapshot for {url}",
        "",
        f"_Crawled {len(docs)} page{'s' if len(docs) != 1 else ''}._",
        "",
    ]

    body = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source") or doc.metadata.get("url") or "Unknown source"
        title = doc.metadata.get("title")
        heading_text = title or source
        body.append(f"## {index}. {heading_text}")
        if source and (not title or title != source):
            body.append(f"[{source}]({source})")
        body.append("")
        if doc.page_content:
            body.append(doc.page_content.strip())
        else:
            body.append("_No content extracted._")
        body.append("")

    return "\n".join(header + body).rstrip() + "\n"


def default_output_path(url: str) -> Path:
    netloc = urlparse(url).netloc or "site"
    safe_netloc = netloc.replace(":", "_")
    return Path(f"{safe_netloc}.md")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("max-depth must be at least 1")
    return parsed


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
    loader = build_loader(
        url=args.url,
        max_depth=args.max_depth,
        exclude_dirs=args.exclude_dirs,
        timeout=args.timeout,
    )
    documents = loader.load()

    if not documents:
        print(f"No documents found for {args.url}", file=sys.stderr)
        return 1

    output_path = args.output or default_output_path(args.url)
    markdown = format_markdown(args.url, documents)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {len(documents)} page(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
