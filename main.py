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
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[".", "\n"])

        self.docs = self.text_splitter.split_documents(self.documents)
        # for doc in self.docs:
        #     print("##############################\n", doc.page_content)
        self.embedding_function = OpenAIEmbeddings()

        self.db = Chroma.from_documents(self.docs, self.embedding_function)
        self.chat_history = []

    def get_response(self, query):
        docs = self.db.similarity_search(query)

        # print results
        messages = [{"role" : "user", "content": query}]
        # messages = []

        num_tokens = len(get_tokens(query)) + len(get_tokens(docs[0].page_content))
        i = 1
        print(f"THE NUMBER OF TOKENS IN QUERY + BEST CONTEXT IS {num_tokens}, looking if I can put chat history")
        pprint(f"THE CHAT HISTORY: {self.chat_history}")
        while i-1 < len(self.chat_history) and num_tokens + len(get_tokens(self.chat_history[-i]["content"])) <= 4000:
            num_tokens += len(get_tokens(self.chat_history[-i]["content"]))
            messages.append(self.chat_history[-i])
            print(f"I've put {self.chat_history[-i]}")
            i += 1
        messages.append({"role" : "system", "content": docs[0].page_content})
        print(f"The current number of tokens after query, chat history and better context is {num_tokens}, will try to put more context")
        i = 1
        while i < len(docs) and num_tokens + len(get_tokens(docs[i].page_content)) <= 4000:
            num_tokens += len(get_tokens(docs[i].page_content))
            messages.append({"role" : "system", "content": docs[i].page_content})
            print(f"Was able to put more context! {i}")
            i += 1
        print(f"Total num of token is {num_tokens}")
        messages = messages[::-1]
        
        pprint(messages)

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        self.chat_history.append({"role" : "user", "content": query})
        self.chat_history.append({"role" : "assistant", "content": str(completion.choices[0].message.content)})

        return (completion.choices[0].message)

if __name__ == "__main__":
    bot = RAG()
    print(bot.get_response(input("ask: ")))