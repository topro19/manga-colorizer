import undetected_chromedriver as uc
import json
import time

options = uc.ChromeOptions()
driver = uc.Chrome(options=options)
driver.get("https://mangadot.net/chapter/37925")
time.sleep(5)

# Fetch manga metadata and chapters list directly in browser session
res = driver.execute_script("""
    return (async () => {
        const r1 = await fetch('https://mangadot.net/api/manga/214/chapters/list');
        const listData = await r1.json();
        
        const r2 = await fetch('https://mangadot.net/api/chapters/37925/images');
        const imgData = await r2.json();
        
        return { chapters: listData, images: imgData };
    })();
""")

print("Images response keys:", res.get('images', {}).keys() if isinstance(res.get('images'), dict) else len(res.get('images', [])))
print("Sample images:", res.get('images'))
print("Total chapters available for this manga:", len(res.get('chapters', [])))
if res.get('chapters'):
    print("Sample chapter entry:", res.get('chapters')[0])
    for c in res.get('chapters')[:5]:
        print("  Ch:", c)

driver.quit()
