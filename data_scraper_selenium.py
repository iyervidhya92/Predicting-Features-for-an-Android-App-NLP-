# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 00:51:47 2017

@author: HakunaMatata
"""

import time
from selenium import webdriver
import csv
import os
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import InvalidSelectorException
from selenium.common.exceptions import ElementNotVisibleException
from selenium.webdriver.common.keys import Keys
import pandas as pd

#profile.set_preference("network.proxy.port", "80")

#from selenium.webdriver.support.ui import WebDriverWait



driver=webdriver.Chrome(executable_path=r'C:\Users\HakunaMatata\Downloads\chromedriver_win32\chromedriver.exe')
current_url = 'https://www.appannie.com/account/login/?_ref=header'


driver.get(current_url)
username = driver.find_element_by_name('username')
username.send_keys("isubramanian3@gatech.edu")

password = driver.find_element_by_name("password")
password.send_keys("diamond9")

driver.find_element_by_id("submit").click()

time.sleep(5)

current_url = "https://www.appannie.com/apps/ios/app/the-wall-street-journal/reviews/?start_date=2010-04-01&end_date=2017-07-20"
driver.get(current_url)
time.sleep(8)

#driver.find_element_by_xpath("""//*[@id="sub-container"]/div[2]/div[2]/div[2]/div[2]/div/div[2]/div/div/div/div[1]/div[1]/div[3]/a[1]""").click()
#driver.find_element_by_xpath("""//*[@id="sub-container"]/div[2]/div[2]/div[2]/div[2]/div/div[4]/div[1]/ul/li[11]/a""").click()

select = Select(driver.find_element_by_xpath("""//*[@id="sub-container"]/div[2]/div[2]/div[2]/div[2]/div/div[2]/div/div/div/div[2]/div[2]/div/div[3]/div/div/div/div[2]/select"""))
select.select_by_visible_text('50')

time.sleep(8)

page_number = driver.find_element_by_xpath('//*[@id="sub-container"]/div[2]/div[2]/div[2]/div[2]/div/div[2]/div/div/div/div[2]/div[2]/div/div[3]/div/div/div/div[2]/form/input')
page_number.clear()
page_number.send_keys("375")
page_number.send_keys(u'\ue007')

time.sleep(8)

reviews=[]
dates=[]
versions=[]
ratings=[]

first_part = '//*[@id="sub-container"]/div[2]/div[2]/div[2]/div[2]/div/div[2]/div/div/div/div[2]/div[2]/div/div[3]/div/div/div/div[1]/div/table/tbody/tr['
second_part = ']/td['
third_part = ']'
fourth_part= '/div'
fifth_part = '/div[2]'
sixth_part = "(//div[@class='original-review'])["

while True:
    j=1
    for i in range(1,50):
         final_xpath1= first_part+str(i)+second_part+str(j)+third_part+fourth_part+fifth_part
         final_xpath2= sixth_part+str(i)+third_part
         final_xpath3= first_part+str(i)+second_part+str(j+2)+third_part
         final_xpath4= first_part+str(i)+second_part+str(j+4)+third_part
         
         rating = driver.find_element_by_xpath(final_xpath1)
         ratings.append(rating.text)
         if not rating:
             break
         review =  driver.find_element_by_xpath(final_xpath2)    
         reviews.append(review.text.encode('utf-8'))
         date = driver.find_element_by_xpath(final_xpath3)
         dates.append(date.text)
         version = driver.find_element_by_xpath(final_xpath4)
         versions.append(version.text)
    try:                                       
        element = driver.find_element_by_xpath('//*[@id="sub-container"]/div[2]/div[2]/div[2]/div[2]/div/div[2]/div/div/div/div[2]/div[2]/div/div[3]/div/div/div/div[2]/button[2]')
        element.send_keys(Keys.PAGE_DOWN)
        element.location_once_scrolled_into_view
        time.sleep(2)
        element.click()
        time.sleep(2)
    except WebDriverException:
        print "WebDriverException"
        break
    except TimeoutException:
        print "TimeoutException"
        break 
    except NoSuchElementException:
        print "NoSuchElementException"
        break
    except InvalidSelectorException:
        print "InvalidSelectorException"
        break
    except ElementNotVisibleException:
        print "ElementNotVisibleException"
        break


filename = "Dataset_CNN_part2.csv" 
filename_text = "Dataset_CNN_part2.txt"
output_path = os.path.expanduser('~')
         
# write output to csv file

with open(output_path+"/"+filename, "w+") as csvfile:  
    writer = csv.writer(csvfile,delimiter = ',',lineterminator = '\n',)
    writer.writerow(['Ratings', 'Reviews', 'Dates', 'Version'])
    writer.writerows(zip(ratings, reviews, dates, versions))

with open(output_path+"/"+filename) as csvfile: 
     df = pd.read_csv(csvfile)
     g = df.groupby('Version')
     for gp in g:
         filename = 'CNN' + gp[0] + '.csv'
         gp[1].to_csv(filename)
         with open(output_path+"/"+filename_text,"w+") as txtfile:
              txtfile.write(filename + '\n')

             
driver.quit()
