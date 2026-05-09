"""Web search tool -- Playwright inside Docker for private scraping.

Spawns a headless Playwright browser inside the Docker sandbox to
fetch and extract text from web pages.  This keeps all network
activity isolated and private.
"""

from __future__ import annotations

from langchain_core.tools import tool

from ojas.tools.docker_exec import run_in_sandbox


_SEARCH_SCRIPT_TEMPLATE = '''\
import sys
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("ERROR: playwright not installed")
    sys.exit(1)

QUERY = """{query}"""

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # Use DuckDuckGo for privacy.
    page.goto(f"https://duckduckgo.com/?q={{QUERY}}")
    page.wait_for_selector(".result", timeout=10000)

    results = page.query_selector_all(".result")
    for i, r in enumerate(results[:5]):
        title_el = r.query_selector(".result__title")
        snippet_el = r.query_selector(".result__snippet")
        title = title_el.inner_text() if title_el else "No title"
        snippet = snippet_el.inner_text() if snippet_el else "No snippet"
        print(f"{{i+1}}. {{title}}")
        print(f"   {{snippet}}")
        print()

    browser.close()
'''


@tool
def search_web(query: str) -> str:
    """Search the web privately using a headless browser in Docker.

    Uses DuckDuckGo via Playwright inside an isolated container.

    Args:
        query: The search query string.

    Returns:
        Top search results as text, or an error message.
    """
    script = _SEARCH_SCRIPT_TEMPLATE.format(query=query.replace('"', '\\"'))
    return run_in_sandbox(
        code=script,
        dependencies=["playwright"],
        timeout=60,
    )


WEB_TOOLS = [search_web]
