import asyncio
from ddgs import DDGS
from playwright.async_api import async_playwright

class WebService:
    def __init__(self):
        self._browser = None
        self._playwright = None
        self._lock = asyncio.Lock() # Prevents race conditions during browser launch

    async def _get_browser(self):
        """Ensures a single Chromium instance is running (Singleton Pattern)."""
        async with self._lock:
            if self._browser is None:
                print("[DEBUG] Launching shared Chromium instance...")
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(headless=True)
            return self._browser

    async def _scrape_url_async(self, url: str, semaphore: asyncio.Semaphore):
        """Internal async method to scrape a single URL using a tab."""
        async with semaphore: # Limit concurrent tabs to prevent bot detection
            print(f"[DEBUG] Scraping: {url}")
            try:
                browser = await self._get_browser()
                
                # Create a new isolated context (Incognito tab)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    viewport={'width': 1280, 'height': 720}
                )
                page = await context.new_page()
                
                await page.goto(url, timeout=15_000, wait_until="domcontentloaded")
                content = await page.inner_text("body")
                
                # Clean up whitespace to save AI tokens
                cleaned_content = " ".join(content.split())
                
                await page.close()
                await context.close()
                
                print(f"[DEBUG] Successfully scraped {len(cleaned_content)} characters from {url}.")
                return cleaned_content[:8000]
            except Exception as e:
                print(f"[DEBUG] Scrape Error for {url}: {str(e)}")
                return f"Failed to scrape {url}: {str(e)}"

    async def _scrape_all_async(self, urls: list[str]):
        """Triggers all scrapes concurrently using a semaphore."""
        if not urls:
            return []
        semaphore = asyncio.Semaphore(3) # Max 3 concurrent tabs
        tasks = [self._scrape_url_async(url, semaphore) for url in urls]
        return await asyncio.gather(*tasks)

    def scrape(self, url: str):
        """Synchronous wrapper for a single URL (to maintain backward compatibility)."""
        if not url:
            return "No URL provided."
        try:
            return asyncio.run(self._scrape_all_async([url]))[0]
        except Exception as e:
            return f"Scrape failed: {str(e)}"

    def scrape_multiple(self, urls: list[str]):
        """New optimized method to scrape multiple URLs in parallel."""
        if not urls:
            return []
        try:
            return asyncio.run(self._scrape_all_async(urls))
        except Exception as e:
            print(f"[DEBUG] Parallel Scrape Error: {str(e)}")
            return [f"Error scraping {url}: {str(e)}" for url in urls]

    def search(self, query: str, max_results: int = 5):
        """Searches DuckDuckGo and returns a list of results."""
        print(f"\n[DEBUG] Starting web search for: '{query}'")
        try:
            with DDGS() as ddgs:
                # Primary Search
                search_results = list(ddgs.text(query, max_results=max_results))
                
                # FALLBACK: If 0 results, try a simplified query
                if not search_results:
                    print(f"[DEBUG] Primary search returned 0 results. Trying fallback...")
                    fallback_query = query.replace("current", "").replace("today", "").replace("right now", "").strip()
                    search_results = list(ddgs.text(fallback_query, max_results=max_results))

                if not search_results:
                    print(f"[DEBUG] All searches returned 0 results.")
                    return [{"title": "No Results", "url": "", "snippet": "No web results found."}]

                print(f"[DEBUG] Found {len(search_results)} results.")
                return [{
                    "title": r.get("title", "No Title"),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                } for r in search_results]
        except Exception as e:
            print(f"[DEBUG] Search Error: {str(e)}")
            return []

# Singleton instance
web_service = WebService()
