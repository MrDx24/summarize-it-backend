from bs4 import BeautifulSoup
import requests


def extractTextFromUrl(link):
    url = link
    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html, 'lxml')
    ans=""
    final=""
    for span_tag in soup.findAll('p'):
        ans+=(span_tag.getText())+" "
        final = ans.replace('\n','')

    return final






