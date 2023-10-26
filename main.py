import os
import openai
import tiktoken
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
    """
    The main RAG chatbot class

    It uses all information given in out.txt to take context for each query
    """
    def __init__(self):
        """
        The constructor of the chatbot

        It is responsible for reading and splitting the data, and also to calculate the embeddings
        for the vector database.

        We are using the Chroma DB as a vector database to use as a retriever for our RAG.
        """

        # Load the data.
        self.loader = TextLoader("out.txt", encoding='utf-8') 
        self.documents = self.loader.load()

        # Split the data into a thoushand-ish characters chuncks.
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[".", "\n"])
        self.docs = self.text_splitter.split_documents(self.documents)

        # Define the openai embedding function to be used in the vector database.
        self.embedding_function = OpenAIEmbeddings()

        #Initialize the vector db with the information about the data and the embedding function.
        self.db = Chroma.from_documents(self.docs, self.embedding_function)
        
        # Last but not least, initialize the chat history as an empty list.
        self.chat_history = []

    def get_response(self, query):
        """
        Gets the response of a query with the OpenAI API sending the query,
        the chat history, and the most relevant chuncks of context from the vector DB.
        """

        # Get the most relevant chuncks from the vector db given the query.
        docs = self.db.similarity_search(query)

        # Initialize the message that will be sent to the OpenAI API.
        # We will send no more than 4 thousands tokens for the API.
        messages = [{"role" : "user", "content": query}]

        # Calculates the number of tokens of the message that will be sent (currently) 
        # that consists of the query and the most relevant chunck for context.
        num_tokens = len(get_tokens(query)) + len(get_tokens(docs[0].page_content))

        # Put as much chat history as possible keeping track of the total number of tokens in the message.
        i = 1
        while i-1 < len(self.chat_history) and num_tokens + len(get_tokens(self.chat_history[-i]["content"])) <= 4000:
            num_tokens += len(get_tokens(self.chat_history[-i]["content"]))
            messages.append(self.chat_history[-i])
            print(f"I've put {self.chat_history[-i]}")
            i += 1
        messages.append({"role" : "system", "content": docs[0].page_content})

        # Put as much context as possible keeping track of the total number of tokens in the message.
        i = 1
        while i < len(docs) and num_tokens + len(get_tokens(docs[i].page_content)) <= 4000:
            num_tokens += len(get_tokens(docs[i].page_content))
            messages.append({"role" : "system", "content": docs[i].page_content})
            i += 1

        # Put the message in the correct message for the OpenAI API.
        messages = messages[::-1]
        
        # Sends the message
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Add the query and the response to the chat history
        self.chat_history.append({"role" : "user", "content": query})
        self.chat_history.append({"role" : "assistant", "content": str(completion.choices[0].message.content)})

        # Returns the response
        return completion.choices[0].message

if __name__ == "__main__":
    bot = RAG()
    print(bot.get_response(input("ask: ")))