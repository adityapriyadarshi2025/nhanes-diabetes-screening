"""Bakes the text and tables that docs/assets/app.js writes into docs/index.html,
so the page reads correctly without JavaScript. Charts and the calculator still
need JavaScript. Run after src.export_web (needs Playwright with Chromium):
    python scripts/prerender.py
"""
import asyncio
import functools
import http.server
import re
import threading
from pathlib import Path

from playwright.async_api import async_playwright

DOCS = Path(__file__).resolve().parents[1] / "docs"
IDS = ["hero-lead", "waffles", "h1-callout", "flow-table", "validation-table", "grid-text", "ladder-text", "yield-table",
       "main-table", "matched-table", "matched-text", "disc-table", "h3-table", "h3-text", "h2-text", "h4-text", "h5-text",
       "worse-text", "worse-table", "missed-table", "missed-text", "conclusion-text"]
SHOWN = ["results-body", "waffle-legend"]


def replace_inner(html, el_id, inner):
    m = re.search(rf'<(\w+)[^>]*\bid="{el_id}"[^>]*>', html)
    tag, start, depth, pos = m.group(1), m.end(), 1, m.end()
    pattern = re.compile(rf"<(/?){tag}\b[^>]*>")
    while depth:
        n = pattern.search(html, pos)
        depth += -1 if n.group(1) else 1
        pos = n.end()
    return html[:start] + inner + html[n.start():]


def unhide(html, el_id):
    return re.sub(rf'(<\w+[^>]*\bid="{el_id}"[^>]*?)\s+hidden(?=[\s>])', r"\1", html)


async def capture(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.route("**/fonts.g*/**", lambda route: route.abort())
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_function("document.querySelectorAll('#conclusion-text p').length > 1")
        inner = {i: await page.locator(f"#{i}").inner_html() for i in IDS}
        await browser.close()
    return inner


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(DOCS)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        inner = asyncio.run(capture(f"http://127.0.0.1:{server.server_port}/index.html"))
    finally:
        server.shutdown()
    path = DOCS / "index.html"
    html = path.read_text()
    for i, content in inner.items():
        html = replace_inner(html, i, content)
    for i in SHOWN:
        html = unhide(html, i)
    path.write_text(html)
    print(f"baked {len(inner)} blocks into {path}")


if __name__ == "__main__":
    main()
