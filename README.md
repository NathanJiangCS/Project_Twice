# Project Twice
This project will be broken down into several parts including a Python Webscraper, Facial Recognition, and hopefully a desktop application or website (still unsure).
Twice, for those who are wondering, is a Korean Pop Group formed by JYP Entertainment consisting of 9 members. Visit http://twice.jype.com/ for more info.
As a Twice fan, I want to keep up to with their activities; however, I am too lazy to take the time to visit the numerous fansites. These fansites (dedicated to one of the nine members) usually compile images of one of the members of the group. I wanted to create a way to collect the data off of multiple sources (fansites and possibly twitter feeds) and compile it into one convinient location.


#Part 1: Web Scraping
I wanted to scrape the various fansites and take the up-to-date images and compile it into one location.
In order to do that, I decided to learn the principles of webscraping and create a simple webscraper in Python using BeautifulSoup4 and URLLib in Python 3.

What does it do:
Simple Python Code that scrapes several of the most popular fansites and fan Twitter feeds for pictures of Twice.
Then it writes the image links in a textfile for other parts of the project to use. The script stores the seen URLs in a list in attempts to prevent duplicate images.

Possible Improvements:
Associate a date with each image so that a chronological sort is possible in the future.

#Part 2:Facial Detection using OpenCv


