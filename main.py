import os
import openai
import tiktoken
from pprint import pprint
import pandas as pd
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader


openai.api_key = os.getenv("OPENAI_API_KEY")

encoding = tiktoken.get_encoding("cl100k_base")

def get_tokens(string):
    return encoding.encode(string)

def tokens_to_str(tokens):
    return [encoding.decode_single_token_bytes(token) for token in tokens]

class RAG():
    def __init__(self):
        self.loader = TextLoader("out.txt", encoding='utf-8')
        self.documents = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"])

        self.docs = self.text_splitter.split_documents(self.documents)
        self.embedding_function = OpenAIEmbeddings()

        self.db = Chroma.from_documents(self.docs, self.embedding_function)


    def get_response(self, query):
        
        docs = self.db.similarity_search(query)

        # print results
        messages = []
        for doc in docs:
            messages.append({"role" : "system", "content": doc.page_content})
        messages = messages[::-1]
        messages.append({"role" : "user", "content": query})

        # print(messages)

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        return (completion.choices[0].message)

if __name__ == "__main__":
    bot = RAG()
    print(bot.get_response(input("ask: ")))