# Project Twice
This project is broken down into several parts including a Python Webscraper, Facial Detection, and Facial Recognition.

Twice, for those who are wondering, is a Korean Pop Group formed by JYP Entertainment consisting of 9 members. Visit http://twice.jype.com/ for more info.

#Part 1: Web Scraping
I wanted to scrape the various Twice fansites and take the up-to-date images and compile it into one location.
In order to do that, I decided to create a simple webscraper in Python using BeautifulSoup4 and URLLib in Python 3. The result a textfile which contains hundreds of links to photos of Twice. For testing purposes, I only scraped 3 fansites which were https://twitter.com/blackpaint96, https://all-twice.com, https://twitter.com/kimdahyun_kr, and https://twitter.com/peachromance. However, there are many more sites out there which can be scraped. In order to ensure that there were no duplicate images (because two fansites could reference the same image), I stored the visited links in array and ensured that the newly-scraped links had not been visited. The results were compiled into links.txt for the other parts of the program to use.

Possible Improvements:
Associate a date with each image so that a chronological sort is possible in the future.
Upgrade the Web Scraper to a Web Crawler. Make it so that it will visit other sites referenced on the fansites.

#Part 2:Facial Detection using OpenCv
The next step was facial detection. Using the Python wrapper for Open CV and some Haar Cascades available at https://github.com/Itseez/opencv/tree/master/data/haarcascades, I was able to create a facial detection program that would go through each photo (linked in the text file) and pick out the faces from the background and the surroundings. This step took the longest as I decided to go through the entire OpenCV documentation and tried all the features that were available. The "learning openCV" directory is the result of my journey to learned what could be achieved using OpenCV Libraries.

#Part 3:Getting a DataBase
In order to do facial recognition, as in identifying someone based on their face, you need pictures of that person which you can train the computer to recognize. Because Twice is a 9-member girl group, I needed a quick way of acquiring a good database of faces. I realized that because I could do facial detection with OpenCV, I technically could create my own database. I would simply save the detected face (detected using OpenCV) into a file and then manually sort the faces based on the members. The "dataset" directory contains the database of faces which I had obtained.

#Part 4: Facial Recognition

