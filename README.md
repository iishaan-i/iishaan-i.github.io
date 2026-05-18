# iishaan-i.github.io

Personal website for Iishaan Inabathini.

## Structure

- `/` (`index.html`) — minimal home page
- `/notes/` — ML paper notes index, with tag filtering and pagination
- `/notes/papers/:title/` — individual paper notes
- `bin/build.py` — converts paper notes from an Obsidian vault into Jekyll posts
- `_data/tag_colors.yml` — optional overrides for tag chip colors

## Authoring paper notes

Paper notes live in an Obsidian vault, not in this repository. A note is
published to the site iff it has `categories: [[Papers]]` **and** `ready: true`
in its frontmatter.

The expected Obsidian frontmatter is:

```yaml
---
categories:
  - "[[Papers]]"
created: 2026-05-11
url: https://arxiv.org/abs/2510.07364
rating: 5
fields:
  - "[[Machine Learning]]"
sub-fields:
  - "[[Mechanistic Interpretability]]"
  - "[[Reasoning Models]]"
ready: true
---
```

Authors and publish dates are intentionally not surfaced on the site.

## Build

Generate `_posts/` from the Obsidian vault:

```bash
pip install -r bin/requirements.txt   # one-time
python bin/build.py
```

The script walks the vault root (non-recursive), filters on `categories=Papers`
and `ready=true`, transforms each note, and writes a Jekyll post per match.
`_posts/` is wiped and rewritten on every run, so un-readying or deleting a
note in Obsidian removes the corresponding post on the next build.

The vault path defaults to `~/Downloads/kepano-obsidian-main`. Override with:

```bash
OBSIDIAN_VAULT=/path/to/vault python bin/build.py
```

## Local preview

```bash
bundle install                        # one-time
bundle exec jekyll serve
```

- Home: `http://localhost:4000/`
- Notes: `http://localhost:4000/notes/`

## Deployment

Commit the generated `_posts/` along with any other changes and push to GitHub.
GitHub Pages serves the site with stock Jekyll (no plugins, no Actions).

## Tag colors

Tags use a deterministic 8-color palette (`slate`, `plum`, `sand`, `moss`,
`rose`, `teal`, `ochre`, `mauve`). A tag's default color is
`palette[sha256(tag) mod 8]`. Override any tag in `_data/tag_colors.yml`:

```yaml
overrides:
  Mechanistic Interpretability: plum
  Reasoning Models: moss
```

## Rating scale

Ratings are integers from 1 to 7 with semantic labels:

| | label |
|---|---|
| 7 | Must read, life changing |
| 6 | Excellent, can't unsee it |
| 5 | Great, a deep read is well worth it |
| 4 | Good |
| 3 | Okay |
| 2 | Bad |
| 1 | Why? |
