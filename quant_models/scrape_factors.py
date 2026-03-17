import requests
from bs4 import BeautifulSoup

url = "https://wmcclinton.github.io/cryptofactors/ui/index.html"
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

for a in soup.find_all('a', href=True):
    if 'CSV' in a.text:
        print(f"Text: {a.text}, Link: {a['href']}")
