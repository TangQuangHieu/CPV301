import json

import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
url = "http://py4e-data.dr-chuck.net/comments_1741217.json"

data = urllib.request.urlopen(url, context=ctx).read().decode()

info = json.loads(data)
count = 0
for item in info['comments']:
    count += int(item['count'])
print(count)