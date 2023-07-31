import streamlit as st
import pandas as pd 
import numpy as np
from Keyword import TextRank4Keyword
from annotated_text import annotated_text
from model import Summary
from transformers import pipeline
import json

#https://www.youtube.com/watch?v=sNhhvQGsMEc
def auto_format(text,keyword):
        modified_text=[]
        random_text = text.split()
        for i in random_text:
            if i in keyword and keyword[i] >= 1:
                modified_text.append((i,'','#fafa'))
            else:
                modified_text.append(i+' ')
        return modified_text

def main(Model=None):
    classifier = pipeline("summarization",model=Model,max_length = 130)
    key_word_extractor = TextRank4Keyword()
    splitted_trans,j = Summary.cleaning(link)
    st.write("Started")
    sample_text = splitted_trans[0]
    #tabs = st.tabs(splitted_trans.keys())
    for i in range(j):
            
        if len(splitted_trans[i][f's_{i}']) > 60:
            with st.spinner('Generating summary...'):
                summary = classifier(splitted_trans[i][f's_{i}'], do_sample=False)[0]['summary_text']
                imp_words = key_word_extractor.analyze(summary)
                
                st.write("From : ",splitted_trans[i]['start'])
                #splitted_trans[i]['summary'] = summary
                #splitted_trans[i]['imp_words'] = imp_words
                print(summary)
                formatted = auto_format(summary,imp_words)
                annotated_text(formatted)
                #st.write(f"Percent Completed : {round(((i+1)/j)*100,2)}")
                st.write("To : ",splitted_trans[i]['end'])
        else:
                classifier2 = pipeline("summarization",min_len=1)
                summary = classifier2(splitted_trans[i][f's_{i}'], do_sample=False)[0]['summary_text']
        st.success(f"Percent Completed : {round(((i+1)/j)*100,2)}")
st.header("""
Youtube Advanced Video Summarizer :sunglasses:
""")

link = st.text_input("Enter YouTube Video link here")

if len(link):
    st.video(link)    
    models = ('Default','facebook/bart-large-cnn')
    selected_model = st.selectbox(label = "Select Model", options = models,placeholder ='Select Model...')
    if selected_model=='Default':
        #classifier = pipeline("summarization",max_length = 130)
        main()
    else:
        #classifier = pipeline("summarization",model=selected_model,max_length = 130)
        main(selected_model)
    st.write("Finished")
