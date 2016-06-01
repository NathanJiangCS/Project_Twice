# Project Twice
This project is broken down into several parts including a Python Webscraper, Facial Detection, and Facial Recognition.

Twice, for those who are wondering, is a Korean Pop Group formed by JYP Entertainment consisting of 9 members. Visit http://twice.jype.com/ for more info.

#Part 1: Web Scraping using URLLib and BeautifulSoup
I wanted to scrape the various Twice fansites and take the up-to-date images and compile it into one location.
In order to do that, I decided to create a simple webscraper in Python using BeautifulSoup4 and URLLib in Python 3. The result a textfile which contains hundreds of links to photos of Twice. For testing purposes, I only scraped 4 fansites which were https://twitter.com/blackpaint96, https://all-twice.com, https://twitter.com/kimdahyun_kr, and https://twitter.com/peachromance. However, there are many more sites out there which can be scraped. In order to ensure that there were no duplicate images (because two fansites could reference the same image), I stored the visited links in array and ensured that the newly-scraped links had not been visited. The results were compiled into links.txt for the other parts of the program to use.

Possible Improvements:
Associate a date with each image so that a chronological sort is possible in the future.
Upgrade the Web Scraper to a Web Crawler. Make it so that it will visit other sites referenced on the fansites.

#Part 2:Facial Detection using OpenCv
The next step was facial detection. Using the Python wrapper for Open CV and some Haar Cascades available at https://github.com/Itseez/opencv/tree/master/data/haarcascades, I was able to create a facial detection program that would go through each photo (linked in the text file) and pick out the faces from the background and the surroundings. This step took the longest as I decided to go through the entire OpenCV documentation and tried all the features that were available. The "learning openCV" directory is the result of my journey to learned what could be achieved using OpenCV Libraries.

#Part 3:Getting a DataBase
In order to do facial recognition, as in identifying someone based on their face, you need pictures of that person which you can train the computer to recognize. Because Twice is a 9-member girl group, I needed a quick way of acquiring a good database of faces. I realized that because I could do facial detection with OpenCV, I technically could create my own database. I would simply save the detected face (detected using OpenCV) into a file and then manually sort the faces based on the members. The "dataset" directory contains the database of faces which I had obtained.

#Part 4: Facial Recognition using FaceRec and Pyfaces
Machine Learning is the study of pattern recognition and computational learning within computers. Using the database of faces, I was able to "train" the computer to recognize the facial patterns of the 9 Twice members. I chose 3 different methods including Eigenfaces(PCA), Fisherfaces, and Linear Discriminant Analysis(LDA). The result of these 3 recognition algorithms were performed on the same image and the results were compared to each other. From experimentation, I concluded that FisherFaces were more accurate than PCA and LDA methods; therefore, when there was inconsistencies among the results, I relied on the FisherFace result. Facial Recognition.py is a script that goes through links.txt and performs facial detection on each image. Then each face within the image undergoes the 3 recognition algorithms and the result is written to the screen.

#Results
Some screenshots of my results are included in the "Results" directory. The facial recognition algorithm was generally successful for frontal faces and less so for side faces. Sometimes, there were false positives resulting in non-faces (such as clothes) being forced through the facial recognition process which resulted in labels being put in some funny places. Other times, the facial recognition would result in someone being mis-identified. I think this is due to the quality of the database that I created; some members had more facial profiles in the database than others which might have skewed the results. 

Possible Improvements:
Haar Cascades are also possible for other things such as eyes and side-face profiles. If one could combine front-face recognition with side-face and eye recognition, it is possible to get much more accurate results; however, getting databases for those things is difficult.
