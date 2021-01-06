#Manning Live Project
#Deep Learning for Finance, Deliverable 1
#David Moskowitz, 1/5/2021
import pdfminer
import textract
import re
import pandas as pd

#Using this because I had no luck with textract
from pdfminer.high_level import extract_text

text=extract_text('sustaintaxonomy.pdf')

#Let's clean this up
regex=re.compile(r"\x0c|\\n{3,}|\.{2,}")
text2=regex.sub("",text)
#I did this step to better see what I was doing, and also I kept tripping over some nasty unicode while saving it to a file.
text2=text2.encode(encoding='ascii', errors='ignore')

#The beginning of the paragraphs
text3=repr(text2).split('\\n\\n')

#For cleaning up the paragraphs
regex=re.compile(r"\\n")

pgraphs=[]
for text4 in text3:
    if len(text4)>=200: #The instructions said 200...I'm wondering if 150 might be better
        text5=regex.sub("",text4) #remove those final extra \ns from ascii
        pgraphs.append(text5)

#building a dictionary from the paragraphs        
paradict={"paragraphs":pgraphs}
#And turning that into a dataframe
df=pd.DataFrame(data=paradict)
