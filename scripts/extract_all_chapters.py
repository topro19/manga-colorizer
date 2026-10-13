import undetected_chromedriver as uc
import time
import json
import re

options = uc.ChromeOptions()
options.add_argument('--disable-blink-features=AutomationControlled')
driver = uc.Chrome(options=options)

try:
    print("Navigating to https://mangadot.net/manga/214 ...")
    driver.get("https://mangadot.net/manga/214")
    for _ in range(10):
        time.sleep(1)
        if "Just a moment" not in driver.title and len(driver.title) > 0:
            break
    print("Page Title:", driver.title)

    # Scroll down to trigger any dynamic lazy load
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    links = driver.execute_script("""
        const anchors = Array.from(document.querySelectorAll('a[href*="/chapter/"]'));
        return anchors.map(a => ({
            href: a.href,
            text: a.innerText.trim()
        }));
    """)
    print("Found chapter links on page:", len(links))

    # Also check Next.js build manifest or page props if present
    next_data = driver.execute_script("""
        if (window.__NEXT_DATA__) {
            return window.__NEXT_DATA__.props;
        }
        return null;
    """)
    if next_data:
        print("Found window.__NEXT_DATA__ props!")
        with open("next_data_props.json", "w") as f:
            json.dump(next_data, f)

    if links:
        print("Sample links:", links[:5])
        with open("dorohedoro_scraped_links.json", "w") as f:
            json.dump(links, f)

finally:
    driver.quit()
