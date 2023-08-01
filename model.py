import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from Keyword import TextRank4Keyword
import streamlit as st
class Summary():
    def __init__(self,max_len = 100,min_len=60):
        self.punc = '''-_?\/'''
        self.max_len = 100
        self.min_len=60
        
    def cleaning(link):
        
        #sample link = 'https://www.youtube.com/watch?v=DxL2HoqLbyA'
        given_input = link
        n = len( 'https://www.youtube.com/watch?v=')
        if given_input[:5] == 'https':
            video_ID = given_input[n:]
        else:
            video_ID = given_input

        detailed_transcript = YouTubeTranscriptApi.get_transcript(video_ID)
        #detailed_transcript = self.retrive_transcript(video_ID)


        N = len(detailed_transcript)
        splitted_transcript = []                
        #print(detailed_transcript[0],detailed_transcript[1])
        i,j =0,0
        while i<N-1:
            start = detailed_transcript[i]['start']
            tmp = ''
            
            while len(tmp)<950 and i < N-1:
                temp = detailed_transcript[i]['text'].replace('-','')
                temp = ' '.join(temp.splitlines())
                tmp += temp
                
                i +=1
                
            #print(i,N)
            end = detailed_transcript[i]['start'] + detailed_transcript[i]['duration']
            dic = {f's_{j}' : tmp, 'start' :start, 'end' : end}
            splitted_transcript.append(dic)
            j +=1
        return [splitted_transcript,j]
    
    def final_summaries(self,link):
        #classifier = pipeline("summarization",model="facebook/bart-large-cnn")
        classifier = pipeline("summarization")
        key_word_extractor = TextRank4Keyword()
        splitted_trans,j = self.cleaning(link)
        print("Start")
        sample_text = splitted_trans[0]
        for i in range(j):
            #print(len(splitted_trans[i][f's_{i}']))
            if len(splitted_trans[i][f's_{i}']) > 60:
                summary = classifier(splitted_trans[i][f's_{i}'], max_length=self.max_len, min_length=self.min_len, do_sample=False)[0]['summary_text']
                imp_words = key_word_extractor.analyze(summary)
                splitted_trans[i]['summary'] = summary
                splitted_trans[i]['imp_words'] = imp_words
            st.write(f'''
            Percent Completed : {round(((i+1)/j)*100,2)}
            ''')
            #if i ==1:
            #    break
        print("Finish")    
        return splitted_trans
