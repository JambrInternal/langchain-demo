# langchain-demo

## URL → Markdown snapshot

This repo includes a small helper script that crawls a URL (recursively) using LangChain's `RecursiveUrlLoader` and writes a single Markdown file containing the extracted pages.

### Setup

```
pip install -r requirements.txt
```

### Usage

```
python url_to_markdown.py https://example.com -d 2 -o example.md
```

Key flags:
- `-d/--max-depth` controls how deep links are followed (minimum/default: 1).
- `-o/--output` sets the Markdown path (defaults to `<domain>.md`).
- `--exclude-dirs` skips path prefixes (e.g. `--exclude-dirs /blog /admin`).
- `--timeout` sets the per-request timeout (seconds).

The output groups each crawled page under its own heading and links back to the source URL.
