"""Extract hyperlinks from markdown content.

Pure function — no external dependencies beyond stdlib (re, urllib).
Designed for independent testing with just a markdown string.
"""

import logging
import re
from urllib.parse import urljoin, urlparse
from typing import List, Dict

logger = logging.getLogger(__name__)

# Regex for markdown links: [anchor_text](url)
_MD_LINK_PATTERN = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')

# File extensions to skip (images, stylesheets, scripts, fonts)
_SKIP_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.ico',
    '.css', '.js', '.woff', '.woff2', '.ttf', '.eot',
    '.pdf', '.zip', '.tar', '.gz',
}


def extract_links_from_markdown(
    markdown: str,
    base_url: str,
) -> List[Dict[str, str]]:
    """Extract all hyperlinks from markdown content.

    Parses markdown for [anchor_text](url) patterns, resolves relative
    URLs against base_url, and captures surrounding context.

    Args:
        markdown: Raw markdown content from a crawled page.
        base_url: The URL of the page the markdown was extracted from.
                  Used to resolve relative links.

    Returns:
        List of dicts, each with keys:
            - url: Resolved absolute URL
            - anchor_text: The link's display text
            - context: Surrounding text (current line + neighbors)
            - source_url: The page this link was found on
    """
    if not markdown or not markdown.strip():
        logger.warning("extract_links: Empty markdown provided")
        return []

    results = []
    seen_urls = set()
    lines = markdown.split('\n')

    for line_idx, line in enumerate(lines):
        for match in _MD_LINK_PATTERN.finditer(line):
            anchor = match.group(1).strip()
            raw_url = match.group(2).strip()

            # Handle Markdown links with titles, e.g., [Text](http://url "Title")
            raw_url = raw_url.split(" ")[0].strip()

            # Resolve relative URLs against base
            resolved = urljoin(base_url, raw_url)
            parsed = urlparse(resolved)

            # Strip fragments (#anchor) so we don't treat different sections of the same page as new pages
            parsed = parsed._replace(fragment="")
            resolved = parsed.geturl()

            # Filter non-http schemes (mailto:, javascript:, etc.)
            if parsed.scheme not in ('http', 'https'):
                continue

            # Filter unwanted file extensions
            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in _SKIP_EXTENSIONS):
                continue

            # Deduplicate
            if resolved in seen_urls:
                continue
            seen_urls.add(resolved)

            # Capture surrounding context (current line + 1 neighbor each side)
            context_start = max(0, line_idx - 1)
            context_end = min(len(lines), line_idx + 2)
            context_lines = lines[context_start:context_end]
            context = ' '.join(l.strip() for l in context_lines if l.strip())

            results.append({
                "url": resolved,
                "anchor_text": anchor,
                "context": context,
                "source_url": base_url,
            })

    logger.info(
        "extract_links: Extracted %d unique links from %s",
        len(results),
        base_url,
    )
    return results
