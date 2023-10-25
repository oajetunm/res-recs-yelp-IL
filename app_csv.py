# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:55:19 2023

@author: OAjetunmobi
"""

#https://medium.com/@woyera/how-to-chat-with-csv-files-using-llama-2-b702182e0868
from tqdm.auto import tqdm
import os
import pinecone
import sys
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
import chainlit as cl
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI
from llama_index.callbacks.base import CallbackManager


os.environ['REPLICATE_API_TOKEN'] = "r8_CfOUSC15HTrlWRFlq5zC16OPxnqWQfb1UsNmZ"

pinecone.init(api_key='5910be35-2c8c-49c0-9e2e-0208a62e9120', environment='gcp-starter')

loader = CSVLoader(file_path="./data/yelp_IL.csv",csv_args = {
                "delimiter": ','}, encoding="utf8")
#                 "quotechar": csv.Dialect.quotechar,)            
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

print("embeddings starting")
embeddings = HuggingFaceEmbeddings()
index_name = "spark"
index = pinecone.Index(index_name)
print("vector starting")
vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)

print("replicate starting")
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.75, "max_length": 3000}
)

print("qa starting")
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)

print("chat history starting")
chat_history = []
while True:
    query = input('Prompt: ')
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))
    

#Creating GUI with tkinter
import tkinter
from tkinter import *


"""def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()"""