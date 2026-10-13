import re
import json

with open('mangadot_solved.html', 'r', encoding='utf-8') as f:
    html = f.read()

print("HTML length:", len(html))

# Check img src
img_tags = re.findall(r'<img[^>]+>', html, re.I)
print(f"Total img tags: {len(img_tags)}")
for tag in img_tags[:15]:
    print("  TAG:", tag)

# Check manga / chapter links
manga_links = re.findall(r'href=["\'](/[^"\']+)["\']', html)
chapters = [l for l in manga_links if 'chapter' in l]
print("Chapter links:", list(dict.fromkeys(chapters))[:20])

# Look for embedded JSON state
scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL | re.I)
for idx, s in enumerate(scripts):
    if len(s) > 100:
        for kw in ['pages', 'images', 'chapter_id', 'cdn', 'chapter']:
            if kw in s:
                print(f"Script #{idx} contains '{kw}' (len: {len(s)})")
                for line in s.splitlines():
                    if any(k in line for k in ['images', 'pages', 'urls', 'chapter']):
                        print("  Line:", line[:150].strip())
                break
