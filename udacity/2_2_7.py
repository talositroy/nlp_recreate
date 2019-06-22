from bs4 import BeautifulSoup
import requests

r = requests.get("https://news.ycombinator.com")
soup = BeautifulSoup(r.text, "html5lib")
print(soup.get_text())
