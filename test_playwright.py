import asyncio
from playwright.async_api import async_playwright
import re

async def test_scrape():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    async with async_playwright() as p:
        print("Launching local Chrome executable...")
        browser = await p.chromium.launch(
            executable_path=chrome_path,
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        target_url = "https://mangadot.net/chapter/37925"
        print(f"Navigating to {target_url}...")
        try:
            await page.goto(target_url, timeout=30000, wait_until="networkidle")
        except Exception as e:
            print("Navigation note:", e)

        title = await page.title()
        print("Page Title:", title)
        
        # Check images on page
        images = await page.evaluate("""() => {
            const imgs = Array.from(document.querySelectorAll('img'));
            return imgs.map(i => i.src || i.dataset.src || i.dataset.original).filter(Boolean);
        }""")
        print(f"Found {len(images)} images!")
        for img in images[:10]:
            print("  Image URL:", img)

        # Check chapter dropdown or links
        chapters = await page.evaluate("""() => {
            const links = Array.from(document.querySelectorAll('a, option'));
            return links.map(l => l.href || l.value).filter(h => h && h.includes('/chapter/'));
        }""")
        print(f"Found {len(chapters)} chapter references!")
        print("Sample chapters:", list(dict.fromkeys(chapters))[:10])

        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_scrape())
