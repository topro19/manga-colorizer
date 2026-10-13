import undetected_chromedriver as uc
import time
import json

options = uc.ChromeOptions()
driver = uc.Chrome(options=options)
driver.get("https://mangadot.net/chapter/37925")

print("Waiting for page and React hydration...")
time.sleep(8)

# Scroll down to trigger lazy loading of images
driver.execute_script("window.scrollTo(0, 1000);")
time.sleep(2)
driver.execute_script("window.scrollTo(0, 3000);")
time.sleep(2)

# Extract images in DOM
images = driver.execute_script("""
    return Array.from(document.querySelectorAll('img')).map(img => ({
        src: img.src,
        dataset: Object.assign({}, img.dataset),
        alt: img.alt
    }));
""")

print(f"Total images found in hydrated DOM: {len(images)}")
for idx, im in enumerate(images[:20]):
    print(f"  Img #{idx}: {im}")

# Extract all links
links = driver.execute_script("""
    return Array.from(document.querySelectorAll('a, button, select, option')).map(el => ({
        tag: el.tagName,
        text: el.innerText,
        href: el.href || el.value
    }));
""")
chapter_links = [l for l in links if l.get('href') and '/chapter/' in str(l.get('href'))]
print(f"Total chapter links in UI: {len(chapter_links)}")
for cl in chapter_links[:10]:
    print("  Chapter link:", cl)

# Also check performance entries / network requests for image CDNs
perf = driver.execute_script("""
    return window.performance.getEntriesByType('resource')
        .map(r => r.name)
        .filter(n => n.includes('.jpg') || n.includes('.png') || n.includes('.webp') || n.includes('chapter'));
""")
print(f"Network resource image requests: {len(perf)}")
for p in perf[:15]:
    print("  Resource:", p)

driver.quit()
