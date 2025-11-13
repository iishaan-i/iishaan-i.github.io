# ML Paper Notes

A minimal Jekyll site for maintaining personal notes on machine learning papers.

## Features

- **Stream-style chronological feed** of paper notes
- **Interactive tag filtering** with AND logic (click multiple tags to filter)
- **Pagination** for browsing large collections
- **Clean, minimal design** inspired by excellent technical writing sites
- **Individual pages** for each paper with full notes

## Getting Started

### Prerequisites

- Ruby (2.7 or higher)
- Bundler

### Local Development

1. Install dependencies:
```bash
bundle install
```

2. Run the Jekyll server:
```bash
bundle exec jekyll serve
```

3. Visit `http://localhost:4000` in your browser

### Adding a New Paper

Create a new file in `_posts/` with the format: `YYYY-MM-DD-paper-title.md`

Front matter template:

```yaml
---
layout: post
title: "Paper Title"
date: YYYY-MM-DD
paper_date: Mon YYYY
authors: Author names
paper_url: https://arxiv.org/abs/...
tags: [Tag1, Tag2]
summary: One sentence summary of the paper and your take.
---

Your notes here in markdown...
```

**Fields:**
- `date`: The date you posted your notes
- `paper_date`: The arxiv v1 publish date (e.g., "Jun 2017")
- Both dates will be displayed on post listings and individual pages

### Available Tags

Current tags (add more as needed in your posts):
- LLM
- Generative Modeling
- Architecture
- RL/Reasoning
- Interpretability

### Deploying to GitHub Pages

1. Create a repository named `<username>.github.io`
2. Push this code to the repository
3. GitHub Pages will automatically build and deploy your site
4. Visit `https://<username>.github.io`

Alternatively, you can deploy to a project repository and it will be available at `https://<username>.github.io/<repository-name>`.

## File Structure

```
.
├── _config.yml          # Jekyll configuration
├── _layouts/            # Page templates
│   ├── default.html     # Base layout
│   └── post.html        # Individual paper page layout
├── _posts/              # Paper notes (markdown files)
├── assets/
│   ├── css/
│   │   └── style.css    # All styles
│   └── js/
│       └── tag-filter.js # Tag filtering logic
├── index.html           # Homepage with paper feed
├── Gemfile              # Ruby dependencies
└── README.md
```

## Customization

### Colors

Tag colors are defined in `assets/css/style.css`. Each tag has its own color scheme with pale backgrounds and darker borders. Modify the `.tag[data-tag="..."]` rules to adjust colors.

### Typography and Spacing

The site uses system fonts for fast loading and native appearance. Adjust font sizes, line heights, and spacing in `style.css`.

### Pagination

Change the number of posts per page in `_config.yml`:

```yaml
paginate: 20  # Change this number
```

## License

Feel free to use this template for your own paper notes site.

