# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:21:09 2018

@author: Ayushi
"""

from  selenium import webdriver
from time import sleep
from bs4 import BeautifulSoup as BS

url = "https://www.homeonline.com/property-for-rent-jaipur/?holsource=paid_search&holmedium=paid_google&gclid=Cj0KCQjwpcLZBRCnARIsAMPBgF3_eZtfDq2LZCtIRdzTXBjCLzrqicZAA09JR2CU8fx8Lx_2QWoUUfMaAjRREALw_wcB"
browser = webdriver.Firefox(executable_path=r"C:\Users\dream\Desktop\geckodriver.exe")
browser.get(url)

def btn_click(browser_con):
    result = browser_con.find_element_by_xpath('/html/body/div[11]/div/div/div/div/div[2]/div[2]/div[2]/a')
    result.click()
    sleep(10)
    
    try:
      html_page = browser_con.page_source
    
    except Exception: 
    #  pass 
        html_page = browser_con.page_source
    return html_page

for i in range(0,4):
    page = btn_click(browser)

soup = BS(page,"lxml")



all_div = soup.find_all('div',class_ = "liststylecon")


#my_div = soup.find_all('div',{"id":"listContent"})
location=[]
area=[]
status=[]

l=[]
bath=[]
deposit=[]
furnished=[]
price = []
for section in all_div:
    info = section.find_all('div',class_="proplisttext")
    for data in info:
        details = data.find_all('div',class_="propdetails")
        for i in details:
            loc = i.findAll('div',class_="col-sm-9")
#location
            raw = loc[0].text.strip()
            n_raw = raw[:raw.find("View on Map")].strip()
            location.append(n_raw)


#bhk

bhk=[]
for section in all_div:
    info = section.find_all('div',class_="row propheading")
    for data in info:
        details = data.find_all('div',class_="col-sm-7 col-md-7")
        for i in details:
            b = i.find('h2')
            b = b.text.strip()
            b=b[0]
            bhk.append(int(b))
