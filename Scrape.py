import urllib
import sys

import urllib.request
from bs4 import BeautifulSoup
db = open('links.txt','w')
images = []
urls = ["https://twitter.com/blackpaint96", "https://twitter.com/peachromance", "https://all-twice.com", "https://twitter.com/kimdahyun_kr"]
for theurl in urls:
    thepage = urllib.request.urlopen(theurl)
    soup = BeautifulSoup(thepage, "html.parser")

    #print(soup.title.text)
    for link in soup.findAll('img'):
        if str(link.get('src')) not in images:
            db.write(str(link.get('src'))+"\n")
            #print (link.get('src'))
            
            images.append(str(link.get('src')))

db.close()
