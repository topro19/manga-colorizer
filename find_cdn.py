import re

with open('mangadot_solved.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Look for image URLs
urls = re.findall(r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp)', html)
print(f"Direct image URLs in HTML: {len(urls)}")
for u in urls[:20]:
    print(" ", u)

matches = re.findall(r'https?://[^\s"\'<>,]+', html)
cdns = [m for m in matches if any(x in m for x in ['cdn', 'storage', 'image', 'manga', 'uploads', 'chapter'])]
print(f"Matching CDN links: {len(cdns)}")
for c in cdns[:25]:
    print("  CDN:", c)
