import undetected_chromedriver as uc
import time
import json

options = uc.ChromeOptions()
options.add_argument('--disable-blink-features=AutomationControlled')
driver = uc.Chrome(options=options)

try:
    driver.get('https://mangadot.net/chapter/37901')

    for _ in range(10):
        time.sleep(1)
        if 'Just a moment' not in driver.title and len(driver.title) > 0:
            break

    # Search for Dorohedoro chapters
    endpoints = [
        'https://mangadot.net/api/manga/214',
        'https://mangadot.net/api/mangas/214',
        'https://mangadot.net/api/mangas/214/chapters',
        'https://mangadot.net/api/chapters?manga_id=214&limit=200',
        'https://mangadot.net/api/manga/dorohedoro'
    ]

    for ep in endpoints:
        res = driver.execute_script("""
            const ep = arguments[0];
            try {
                const resp = await fetch(ep);
                if (!resp.ok) return { status: resp.status };
                return await resp.json();
            } catch(e) {
                return { error: e.toString() };
            }
        """, ep)
        print(ep, "->", list(res.keys()) if isinstance(res, dict) else f"List length: {len(res)}")
        if isinstance(res, list) or (isinstance(res, dict) and ('chapters' in res or 'data' in res)):
            with open('dorohedoro_chapters_found.json', 'w') as f:
                json.dump(res, f)
            print("Successfully saved chapters to dorohedoro_chapters_found.json!")
            break
finally:
    driver.quit()
