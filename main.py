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
# if __name__ == '__main__':
    

encoding = tiktoken.get_encoding("cl100k_base")

def get_tokens(string):
    return encoding.encode(string)

def tokens_to_str(tokens):
    return [encoding.decode_single_token_bytes(token) for token in tokens]

def get_data_from_txt(path_to_file):
    f = open(path_to_file, "r", encoding="utf-8")
    return "".join([line for line in f])

data = get_data_from_txt("out.txt")
loader = TextLoader("out.txt", encoding='utf-8')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"])

docs = text_splitter.split_documents(documents)
embedding_function = OpenAIEmbeddings()

db = Chroma.from_documents(docs, embedding_function)

query = input("Pergunta: ")
docs = db.similarity_search(query)

# print results
messages = []
for doc in docs:
    messages.append({"role" : "system", "content": doc.page_content})
messages = messages[::-1]
messages.append({"role" : "user", "content": query})

print(messages)

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
)

print(completion.choices[0].message)

# print(docs[0].page_content)

# embeddings = calculate_embeddings(chuncks, recalculate=True)

# print(embeddings)