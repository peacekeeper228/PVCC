# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:04:37 2022

@author: Александр
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import json

def findComp(soup):
    r = soup.find_all("div", {"class": "content"})
    k=r[0].text[9:]
    st=' '.join(k.split())
    st=st[:-234]
    return st
def findSAPR(soup):
    r=soup.find_all(id="article")
    try:
        k=r[0].text
    except IndexError:
        return -1
    else:
        st=' '.join(k.split())
        st[:-104]
        lastWord=max([st.rfind('.'),st.rfind('!'),st.rfind('?')])
        return st[:lastWord+1]
def findCompress(soup):
    r = soup.find_all("main", {"class": "content"})
    k=r[0].text
    st=' '.join(k.split())
    k=st.find('(adsbygoogle ')
    st=''.join([st[:k],st[k+51:]])
    #st.replace('(adsbygoogle = window.adsbygoogle || []).push({});','')
    return st[:-74]

# 22000
lis = [i for i in range (26300,26370)]

broken=[]
p404=[]
l=[]
empty=[]
for i in lis:
    url='https://sapr.ru/article/'+str(i)
    req=requests.get(url)
    s=req.text
    soup=BeautifulSoup(s, "html5lib")
    time.sleep(0.5)
    if 'Извините, документ не найден' in req.text:
        print(i, ' is 404')
        p404.append(url)
        continue
    if 'compuart' in req.url:
        k=findComp(soup)
        l.append(k)
        print(i,' parsed CompArt')
    elif 'sapr' in req.url:
        k=findSAPR(soup)
        if k!=-1:
            l.append(k)
            print(i,' parsed SAPR')
        else:
            empty.append(url)
            print(i, ' SAPR empty')
    elif 'compress' in req.url:
        k=findCompress(soup)
        l.append(k)
        print(i,' parsed compress')
    else:
        broken.append(url)
        print(i,' skipped')
        
with open('data_SAPR.json', 'w',encoding='utf-8') as f:
    json.dump(l, f)

def writeInFile(linkList,file):
    f=open(file,'w')
    for item in linkList:
        f.write("%s\n" % item)

writeInFile(broken,'lab4/brokenlinks_1.txt')
writeInFile(p404,'lab4/pages404_1.txt')
writeInFile(empty,'lab4/emptyLinks_1.txt')
