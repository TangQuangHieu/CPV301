import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
import ssl

api_key = False
# If you have a Google Places API key, enter it here
# api_key = 'AIzaSy___IDByT70'
# https://developers.google.com/maps/documentation/geocoding/intro


# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

#while True:
url = "http://py4e-data.dr-chuck.net/comments_1741216.xml"
#if len(address) < 1: break

print('Retrieving', url)
uh = urllib.request.urlopen(url, context=ctx)

data = uh.read()
tree = ET.fromstring(data)

counts = tree.findall(".//count")
total = 0
for count in counts:
    total += float(count.text)
print(total)

