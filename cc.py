import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import BeautifulSoupLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Scrape data from the URL
url = "https://brainlox.com/courses/category/technical"
response = requests.get(url)
html_content = response.text

soup = BeautifulSoup(html_content, "html.parser")
courses = soup.find_all('div', class_='course-card')

documents = []
for course in courses:
    title = course.find('h4').text.strip()
    description = course.find('p').text.strip()
    document_content = f"{title}\n{description}"
    documents.append(document_content)

document_texts = "\n".join(documents)
loader = BeautifulSoupLoader(document_texts)
documents = loader.load()

# 2. Create embeddings and store in a vector store
embeddings_model = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
doc_embeddings = embeddings_model.embed_documents(docs)

vector_store = FAISS.from_documents(docs, doc_embeddings)
vector_store.save_local("faiss_store")

# 3. Create a Flask RESTful API
app = Flask(__name__)
api = Api(app)

# Load the saved vector store
vector_store = FAISS.load_local("faiss_store", embeddings_model)

qa_chain = load_qa_chain(vector_store, embeddings_model)

class Chatbot(Resource):
    def post(self):
        data = request.get_json()
        user_input = data.get('query')
        answer = qa_chain.run(input_document=user_input)
        return jsonify({"response": answer})

api.add_resource(Chatbot, '/chatbot')

if __name__ == '__main__':
    app.run(debug=True)
