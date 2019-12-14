# Adapted from example in Ch.3 of "Web Scraping With Python, Second Edition" by Ryan Mitchell

import re
import csv
import requests
from bs4 import BeautifulSoup
import os, sys
pages = set()
from newspaper import Article

def get_links(news_url, dpt):

	csvFile = csv.writer(open('dataset.csv', 'a'))
	# "news","family","education","sex-relationship","entertainment","politics","sports",
	# https://punchng.com/topics/spice/page/2/
	categories = ["business"]
	start_page = 3
	num_page_to_crawl = 71


	global pages
	pattern = re.compile("^(/)")
	# pattern = re.compile("http://www")
	#   html = requests.get(f"your_URL{page_url}").text # fstrings require Python 3.6+

	# writer = csv.writer(open("dataset", 'w'))
	

	if not os.path.exists("datasets/"):
		os.mkdir("datasets/")

	for category in categories:

		for page_num in range(start_page, num_page_to_crawl):
			page_url = news_url + "topics/" + category + "/page/"+ str(page_num)
			# print(category)
			print(" >>>> ", page_url)
			
			html = requests.get(page_url).text

			soup = BeautifulSoup(html, "html.parser")

			# res = soup.find_all("a")
			news_link = 1
			for link in soup.find_all("a"):
		
				# print("")
				# print("=======================================")
				if "href" in link.attrs:

					new_page = link.attrs["href"]
					
					if len(new_page) < 40:
						continue

					if not "/topics/" in new_page:
						# print(" <<>> ", new_page)

						header = new_page.split("/")
						header = header[len(header) -1]
						# print(" <<<  ",header)

						content_html = requests.get(new_page).text
						content_soup = BeautifulSoup(content_html, "html.parser")
						header = content_soup.find("h1")
						
						if not header:
							continue

						article = Article(new_page)
						article.download()
						article.parse()
						
						
						news_header = header.getText()
						news_content = article.text

						if not os.path.exists("datasets/" + category):
							os.mkdir("datasets/"+category)
						
						file = open( "datasets/"+category+"/" + str(page_num) + "-" +  str(news_link)+ ".txt","w")
						file.write( news_header + "\n" + news_content)
						file.close() 

						print(category , " -> ", news_link , " >> ")

						# print(news_link ," <<< ",  news_header)
						news_link = news_link + 1

						

			# print("=======================================================")
	print(" >>> Done with crawler >> ")
	
# get_links("https://bbc.co.uk/", 1)
get_links("https://punchng.com/", 1)