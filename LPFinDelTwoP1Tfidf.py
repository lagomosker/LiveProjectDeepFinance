#Manning Live Project
#Deep Learning for Finance, Deliverable 1
#David Moskowitz, 1/5/2021

"""
Deliverable for part2. New code begins with "#New code begins here"
Platform issues are the reason for a separate file for the Word2Vect model, even so I got an error in  
    print(f"Word 'electricty' appeared {model.wv.get_vecattr('electricity', 'count')} times in the training corpus.")

Notes:
0. I plan to go back to this, but I didn't want to get to far behind and wanted to see competent work to help guide me
1. With this TFDIF method, I'm unsure if the results are due to my poor implementation, questionable data, or both. 
    This is especially apparent in my attempt at filling in arrays of word counts and the ham-fisted attempt to gague
    similarity seeking the smallest magnitude of the linear kernel
2. All of the latter text files are so I can see what's generated
3. I added a one word question "Sludge" to the questions.txt file and noticed that didn't even give me anything of note until I upped the epoch count in the training.
4-100. Slowly learning just how much more I have to learn: next step, learning what I need to learn ASAP
"""

import pdfminer  
import textract
import re
import pandas as pd
import numpy
import copy
import numpy.linalg
import smart_open

#Using this because I had no luck with textract
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from numpy.linalg import norm
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
output=open('SustainParagraphs.txt', 'w')
for text4 in text3:
    if len(text4)>=200: #The instructions said 200...I'm wondering if 150 might be better
        text5=regex.sub("",text4) #remove those final extra \ns from ascii
        pgraphs.append(text5)
        output.writelines(text5+"\n")

output.close()
#building a dictionary from the paragraphs        
paradict={"paragraphs":pgraphs}
#And turning that into a dataframe
df=pd.DataFrame(data=paradict)


#New code begins here
questions=[]
with open ('Questions.txt', 'r') as reader:
    print(reader)
    question=reader.readline() 
    while question!='':
        print(question)
        questions.append(question.rstrip())
        question=reader.readline()
reader.close()


print(questions)
#quescount_file.write(repr(word_list))
quescounts=[]

#TFDIF model
index=0
for quest in questions:
    quescount_file=open('quescounts' +str(index) +'.txt','w')
    #vectorizerQuestions=TfidfVectorizer()
    p_index=0
    top_answer=0
    best_match=0
        
    for pgraph in pgraphs:
        vectorizerSustain=TfidfVectorizer()
        sustainVector=vectorizerSustain.fit_transform(pgraph.split())
        #print ("Ustain vector" , sustainVector.shape)
        word_list=vectorizerSustain.get_feature_names()
        ques_string=quest
    #questionVector=vectorizerQuestions.fit_transform(quest.split())
        ques_word_array=numpy.zeros((1,sustainVector.shape[1]),dtype=int)
    #   print(repr(word_list))
        for questword in quest.split(): #vectorizerQuestions.get_feature_names():
            if questword in word_list:
                ques_string+= " " + questword + str(word_list.index(questword)) 
                #print(ques_string + str(word_list.index(questword)))
                ques_word_array[0][word_list.index(questword)]+=1
        answers=linear_kernel(sustainVector, ques_word_array)
        temp=numpy.linalg.norm(answers)
        if p_index>0: #dealing with the initial case
            if temp>top_answer:
                best_match=p_index
                top_answer=temp
        p_index+=1
    quescount_file.write(quest + " ans score: " +  str(top_answer) +"\n\n" + pgraphs[best_match])# + " " +ques_string + repr(ques_word_array) + " " +  "ans" + repr(answers) +"\n" )
    index+=1
    quescount_file.close

