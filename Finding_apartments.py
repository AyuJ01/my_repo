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

for i in range(0,42):
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
#location
for section in all_div:
    info = section.find_all('div',class_="proplisttext")
    for data in info:
        details = data.find_all('div',class_="propdetails")
        for i in details:
            loc = i.findAll('div',class_="col-sm-9")
            
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
            if len(b)==0:
                bhk.append(int('1'))
            else:
                b=b[0]
                bhk.append(int(b))

#area
area=[]
for section in all_div:
    info = section.find_all('div',class_="proplisttext")
    for data in info:
        details = data.find_all('div',class_="propdetails")
        for i in details:
            loc = i.findAll('div',class_="col-sm-9")
            
            raw = loc[1].text.strip().split()
            raw = raw[0]
            l=[]
            for i in raw:
                if i.isdigit():
                    l.append(i)
                    
            s=''.join(l)
            if len(s) == 0:
                area.append(int('800'))
            else:    
                area.append(int(s))

#price
price=[]
for section in all_div:
    info = section.find_all('div',class_="proplisttext")
    for data in info:
        details = data.find_all('div',class_="row propheading")
        for i in details:
            loc = i.find('div',class_="pricesty")
            raw = loc.text.strip()
            raw = raw.replace(",","")
            raw = raw.split()
            if len(raw) == 3 and raw[1].lower() in ['lac','lacs','lakh','lakhs']:
                price.append(int(float(raw[0])*100000))
            else:
                price.append(int(float(raw[0])))
            
            
        
#Making the dataframe
import pandas as pd

df1 = pd.DataFrame(location,columns = ['Location'])
df1['Area'] = area
df1['BHK'] = bhk
df1['price'] = price
df1.to_csv("house_price_3.csv",index=False)

df1 = pd.get_dummies(df1, columns=["Location"])
features = df1.drop("price",axis=1).values
labels = df1["price"].values
        
#Label Encoding
#from sklearn.preprocessing import LabelEncoder
#encoder = LabelEncoder()
#features["Location"] = encoder.fit_transform(features["Location"])

#features scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = sc.fit_transform(features)

#splitting the dataset
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)


#Fit Linear Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(100,random_state=0)
regressor.fit(features_train,labels_train)
pred = regressor.predict(features_test)
score = regressor.score(features_test,labels_test)




