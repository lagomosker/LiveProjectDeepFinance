"""
Third deliverable.
The more I read the instructions, the more I realized I was not only not on the right track. I was on a completely different rail line.
This is just me playing around with models, and letting me see what you did. Starting to understand the concepts, will have more input soon.  
Right now, I'm just mixing and matching models and different Question/Best paragraph selections from week 2.

"""


import pickle
import torch
import transformers
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

datasets=[("bertlargeUC","bert-large-uncased-whole-word-masking-finetuned-squad"),("robertabased", "roberta-base"), ("distilbertUC","distilbert-base-uncased-distilled-squad")]
questions = [
    ["Does a company need to keep track of the carbon intensity of the electricity?"],
    ["What metric is used for evaluating emission?"],
    ["How does one get to net-zero emissions economy?"],
    ["What is net-zero emissions economy?"],
    ["How can carbon emission of the processes of cement clinker be reduced?"],
    ["How is the Weighted Cogeneration Threshold calculated?"],
    ["What is carbon capture and sequestration?"],
    ["What stages does CCS consist of?"],
    ["What should be the average energy consumption of a water supply system?"],
    ["What are sludge treatments?"],
    ["How is the process of anaerobic digestion?"],
    ["What is considered Zero direct emission vehicles?"]
]

#These two files were made from my own cleansing of the PDF data
#Followed by cut and paste milestone #2 from Matteus Tahana to insure things didn't go completely off the 
with open("qcontextdoc2vec.dat", "rb") as input_file:
    doc2vec_tuple= pickle.load(input_file)

with open("tdifquestions.dat", "rb") as input_file:
    tfidf_tuple= pickle.load(input_file)


score_dictionary={}

for question in questions:
    score_dictionary[str(question)]={'doc2vec':[], 'tfidf':[]}

print(str(score_dictionary))

#First let's see how the different models work on different question/paragraphs in different situations from milestone2
for dataset in datasets:
    nlp=pipeline("question-answering", dataset[1])
    for count, question in enumerate(questions):
        ans=nlp(question=doc2vec_tuple[count][0],context=doc2vec_tuple[count][1])
        score_dictionary[str(question)]['doc2vec'].append((dataset[0],ans))
        ans=nlp(question=tfidf_tuple[count][0],context=tfidf_tuple[count][1])
        score_dictionary[str(question)]['tfidf'].append((dataset[0],ans))

print(str(score_dictionary))

#Rank them, save them. 
result_file=open("Pipeline.txt", 'w')
top_handler=open("topscores.txt",'w')
top_scores=[]
for question in questions:
    highest_score=None
    result_file.write(str(question)+"\n")
    top_handler.write(str(question)+"\n")
    responses=score_dictionary[str(question)]
    result_file.write("\tDoc2Vec\n")
    doc2vec_answers=responses['doc2vec']
    for answers in doc2vec_answers:
        if highest_score==None:
            highest_score=('d2v '+str(answers[0]),answers[1])
        elif highest_score[1]["score"]<=answers[1]["score"]:
            highest_score=('d2v '+str(answers[0]),answers[1])
        result_file.write("\t\t" + answers[0] + ":" + answers[1]["answer"] + " score:" +str(round(answers[1]["score"],3))+"\n")
    tfidf_answers=responses['tfidf']
    result_file.write("\tTfIDF\n")
    for answers in tfidf_answers:
        if highest_score[1]["score"]<=answers[1]["score"]:
            highest_score=('tfdif '+str(answers[0]),answers[1])
        result_file.write("\t\t" + answers[0] + ":" + answers[1]["answer"] + " score:" +str(round(answers[1]["score"],3))+"\n")
    result_file.write("\n")
    top_handler.write("\t\t" + highest_score[0] + ":" + highest_score[1]["answer"] + " score:" +str(round(highest_score[1]["score"],3))+"\n")
result_file.close()
top_handler.close()
