#!/usr/bin/env python3
"""Convert paper notes from an Obsidian vault into Jekyll posts.

Walks the vault root (non-recursively) looking for `.md` files that have
`categories: [[Papers]]` and `ready: true` in their YAML frontmatter.
For each matching note, emits a corresponding file under `_posts/`.

The destination `_posts/` directory is wiped at the start of every run so
that un-readying or deleting an Obsidian note removes it from the site.

Configuration:
    OBSIDIAN_VAULT  Path to the vault root.
                    Default: ~/Downloads/kepano-obsidian-main
"""

from __future__ import annotations

import hashlib
import html
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "PyYAML is required. Install with: pip install -r bin/requirements.txt\n"
    )
    sys.exit(1)


PALETTE = ["slate", "plum", "sand", "moss", "rose", "teal", "ochre", "mauve"]

RATING_LABELS = {
    7: "Must read, life changing",
    6: "Excellent, can't unsee it",
    5: "Great, a deep read is well worth it",
    4: "Good",
    3: "Okay",
    2: "Bad",
    1: "Why?",
}

DEFAULT_VAULT = "~/Downloads/kepano-obsidian-main"

WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")
REFERENCES_HEADING_RE = re.compile(r"^#\s+References\s*$\n?", re.MULTILINE)
HEADING_RE = re.compile(r"^(#{1,5})(\s+\S)", re.MULTILINE)
FENCED_CODE_RE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)
MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$([^\$\n]+?)\$", re.DOTALL)


def site_root() -> Path:
    return Path(__file__).resolve().parent.parent


def strip_wikilink_scalar(s: object) -> object:
    """Strip wrapping `[[ ... ]]` from a single frontmatter string value."""
    if not isinstance(s, str):
        return s
    m = WIKILINK_RE.fullmatch(s.strip())
    if not m:
        return s
    inner = m.group(1)
    if "|" in inner:
        inner = inner.split("|", 1)[1]
    return inner


def strip_wikilinks_in_text(text: str) -> str:
    """Replace `[[foo]]` and `[[foo|bar]]` with `foo` / `bar` everywhere."""

    def repl(match: re.Match[str]) -> str:
        inner = match.group(1)
        if "|" in inner:
            return inner.split("|", 1)[1]
        return inner

    return WIKILINK_RE.sub(repl, text)


def parse_frontmatter(content: str) -> tuple[dict | None, str]:
    if not content.startswith("---"):
        return None, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None, content
    try:
        fm = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None, content
    if not isinstance(fm, dict):
        return None, content
    return fm, parts[2].lstrip("\n")


def is_paper(fm: dict) -> bool:
    cats = fm.get("categories")
    if not isinstance(cats, list):
        return False
    return any(strip_wikilink_scalar(c) == "Papers" for c in cats)


def to_iso_date(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip()).isoformat()
        except ValueError:
            return None
    return None


def slugify(title: str) -> str:
    s = title.lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_-]+", "-", s)
    return s.strip("-") or "untitled"


def tag_color(tag: str, overrides: dict[str, str]) -> str:
    if tag in overrides and overrides[tag] in PALETTE:
        return overrides[tag]
    h = hashlib.sha256(tag.encode("utf-8")).digest()[0]
    return PALETTE[h % len(PALETTE)]


def load_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    overrides = data.get("overrides") if isinstance(data, dict) else None
    if not isinstance(overrides, dict):
        return {}
    return {str(k): str(v) for k, v in overrides.items()}


def collect_tags(fm: dict) -> list[str]:
    tags: list[str] = []
    for key in ("fields", "sub-fields"):
        values = fm.get(key) or []
        if not isinstance(values, list):
            continue
        for value in values:
            cleaned = strip_wikilink_scalar(value)
            if isinstance(cleaned, str):
                cleaned = cleaned.strip()
            if cleaned and cleaned not in tags:
                tags.append(cleaned)
    return tags


def protect_math_for_kramdown(body: str) -> str:
    """Replace math spans with HTML placeholders carrying the LaTeX source.

    Kramdown otherwise mangles LaTeX in three ways:
      - `$$...$$` blocks get rewritten to `\\(...\\)`, so KaTeX (which we've
        configured for `$$/$`) doesn't pick them up.
      - Inside `\\begin{align}` blocks, the `&` column separators and `\\\\`
        row separators trigger kramdown's table parser and emphasis escaping.
      - Inline `$\\pi(\\cdot|x_1)$` spans with `|` characters on adjacent
        lines also trigger the table parser.

    `{::nomarkdown}` would in theory fix this, but kramdown's table parser
    runs before extensions are recognized, so the inline-extension markers
    leak into the output. The robust fix is to emit a placeholder
    `<span class="math-render" data-math="..." data-display="...">` for
    each math span. The LaTeX lives in an HTML attribute, which kramdown
    leaves untouched, and a small client script renders the placeholders
    with KaTeX once the page loads.
    """

    def repl(match: re.Match[str]) -> str:
        block = match.group(1)
        if block is not None:
            tex = block
            display = "block"
        else:
            tex = match.group(2)
            display = "inline"
        # Collapse newlines so the math survives as a single-line HTML
        # attribute. LaTeX is whitespace-insensitive (row breaks in `align`
        # are denoted by `\\`, not newlines), so this is safe.
        tex = re.sub(r"\s+", " ", tex).strip()
        escaped = html.escape(tex, quote=True)
        span = (
            f'<span class="math-render" data-display="{display}" '
            f'data-math="{escaped}"></span>'
        )
        # Force display math onto its own paragraph. Without surrounding
        # blank lines, kramdown treats a `$$...$$` that immediately
        # follows a list item as a lazy continuation and nests the math
        # (and any text after it) inside the previous `<li>`, picking up
        # the bullet's indentation.
        if display == "block":
            return f"\n\n{span}\n\n"
        return span

    body = MATH_RE.sub(repl, body)
    # Collapse runs of 3+ newlines created by the block-math padding.
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body


def downshift_headings(body: str) -> str:
    """Shift `#` -> `##`, etc., everywhere except inside fenced code blocks.

    The paper title is the page's single `<h1>`; section headings inside the
    note should start at `<h2>`. Capped at five `#` so `######` stays valid.
    """

    def shift(text: str) -> str:
        return HEADING_RE.sub(lambda m: "#" + m.group(1) + m.group(2), text)

    pieces: list[str] = []
    last = 0
    for match in FENCED_CODE_RE.finditer(body):
        pieces.append(shift(body[last:match.start()]))
        pieces.append(match.group(0))
        last = match.end()
    pieces.append(shift(body[last:]))
    return "".join(pieces)


def transform_body(body: str) -> str:
    body = strip_wikilinks_in_text(body)
    body = REFERENCES_HEADING_RE.sub("", body)
    body = downshift_headings(body)
    body = protect_math_for_kramdown(body)
    return body


def build_post(
    md_path: Path,
    fm: dict,
    body: str,
    overrides: dict[str, str],
) -> tuple[str, str] | None:
    title = md_path.stem
    created = to_iso_date(fm.get("created"))
    if not created:
        sys.stderr.write(
            f"skipping {md_path.name}: missing or invalid `created` date\n"
        )
        return None

    rating = fm.get("rating")
    if not isinstance(rating, int) or rating not in RATING_LABELS:
        rating = None

    tags = collect_tags(fm)
    tag_data = [{"name": t, "color": tag_color(t, overrides)} for t in tags]

    out_fm: dict = {
        "layout": "post",
        "title": title,
        "date": created,
    }
    if rating is not None:
        out_fm["rating"] = rating
        out_fm["rating_label"] = RATING_LABELS[rating]
    if tags:
        out_fm["tags"] = tags
        out_fm["tag_data"] = tag_data

    url = fm.get("url")
    if isinstance(url, str) and url.strip():
        out_fm["paper_url"] = url.strip()

    new_body = transform_body(body)

    frontmatter_yaml = yaml.safe_dump(
        out_fm,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    output = f"---\n{frontmatter_yaml}---\n\n{new_body}"

    slug = slugify(title)
    filename = f"{created}-{slug}.md"
    return filename, output


def main() -> int:
    vault_env = os.environ.get("OBSIDIAN_VAULT", DEFAULT_VAULT)
    vault = Path(vault_env).expanduser()
    if not vault.is_dir():
        sys.stderr.write(f"Obsidian vault not found at {vault}\n")
        return 1

    root = site_root()
    posts_dir = root / "_posts"
    overrides_path = root / "_data" / "tag_colors.yml"

    overrides = load_overrides(overrides_path)

    posts_dir.mkdir(exist_ok=True)
    for old in posts_dir.glob("*.md"):
        old.unlink()
    for old in posts_dir.glob("*.markdown"):
        old.unlink()

    written = 0
    skipped = 0
    for md_path in sorted(vault.glob("*.md")):
        with md_path.open("r", encoding="utf-8") as f:
            content = f.read()
        fm, body = parse_frontmatter(content)
        if fm is None:
            continue
        if not is_paper(fm):
            continue
        if not fm.get("ready"):
            skipped += 1
            continue

        result = build_post(md_path, fm, body, overrides)
        if result is None:
            continue
        filename, output = result

        out_path = posts_dir / filename
        with out_path.open("w", encoding="utf-8") as f:
            f.write(output)
        written += 1

    print(f"Wrote {written} posts to {posts_dir}")
    if skipped:
        print(f"({skipped} papers in vault are not yet `ready: true`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
