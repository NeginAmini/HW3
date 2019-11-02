# HW3

import requests, urllib
url = 'https://raw.githubusercontent.com/CriMenghini/ADM/master/2019/Homework_3/data/movies1.html'
response = requests.get(url)

from bs4 import BeautifulSoup
soup = BeautifulSoup(response.text, 'lxml')

url_list = []
for Url in soup.find_all('a'):
  url_list.append(Url.get('href'))

d ={}  
for i in url_list:  
  html_wiki = urllib.request.urlopen(i).read()
  soup = BeautifulSoup(html_wiki ,'html.parser')  
  for x,y in  zip(soup.table.find_all('th') , soup.table.find_all('td')):
  d[x.text] = y.text
