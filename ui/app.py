from flask import Flask, render_template, request
from flask_cors import CORS

import base64
import utils

# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# https://huggingface.co/blog/sentence-transformers-in-the-hub
from sentence_transformers import SentenceTransformer

# open AI init
import openai
openai.api_key = "<your-open-ai-key>" # openai api_key.

# Elastic Search cloud init
from elasticsearch import Elasticsearch

# Elastic Search Variables
HOST_IP = "localhost"
PORT = 9200
CA_CERT_PATH = "/Users/arskayal/elasticsearch-8.9.2/config/certs/http_ca.crt"
USERNAME="elastic" # by default
PASSWORD="e5FmVlxHa568xrO33guk"
index_name = "medium-article"

# NLP models
embedding_model_name = "sentence-transformers/sentence-t5-base"
keyword_extraction_model_name = "ml6team/keyphrase-extraction-kbir-inspec"

# Load NLP models
embedding_model = SentenceTransformer(embedding_model_name)
extractor = utils.KeyphraseExtractionPipeline(model=keyword_extraction_model_name)

app = Flask(__name__)
cors = CORS(app) # to avoid cors error

print("dependencies loaded . . .")

@app.route('/')
def index():
    return render_template('index.html')

def connect_elastic():
    es_conn = Elasticsearch(
        [{
            "host": HOST_IP,
            "port": PORT
        }],
        http_auth=(USERNAME, PASSWORD),
        verify_certs=True,
        use_ssl=True,
        ca_certs=CA_CERT_PATH
    )
    if es_conn.ping():
        print("Connected to elasticsearch ... ")
    else:
        print("Elasticsearch connection error ...")
    return es_conn

client = connect_elastic()

def get_transcript(encoded_audio):
    '''
    Save encoded audio to a file for reading
    '''
    # Decode the base64 data
    decoded_audio = base64.b64decode(encoded_audio)

    # Write the decoded audio to a file
    with open('decoded_audio.wav', 'wb') as f:
        f.write(decoded_audio)
        
    audio_file= open('decoded_audio.wav', "rb")

    transcript = openai.Audio.transcribe("whisper-1", audio_file, language="en")
    print(f"Transript: {transcript['text']}\n")
    return transcript['text']


def semantic_search(es_client, index_name, query_text, thresh=1.6, top_n=10):

    token_vector = utils.embed_text(embedding_model, query_text)
    count = es_client.cat.count(index=index_name, params={"format": "json"})[0]["count"]
    print(f'Showing recommendations from {count} documents ...')
    if not es_client.indices.exists(index=index_name):
        return "No records found"

    query2 = {
        "size": top_n,
        # "_source": "title",
        "query": {
            "bool": {
                "must": []
            }
        },
        "knn": {
            "field": "embedding_vectors",
            "query_vector": token_vector,
            "k": 10,
            "num_candidates": 100
        }
    }
    result = es_client.search(index=index_name, body=query2)
    total_match = len(result['hits']['hits'])
    print("Total Matches: ", str(total_match))

    print()
    counter = 0
    output_list = []
    # if len(query_text.split()) > 2:
    keyphrases = extractor(query_text)
    # else:
    #     keyphrases = [w for w in query_text.split() if not w.lower() in stop_words]
    if total_match > 0:
        for hit in result['hits']['hits']:
            counter += 1
            output_json = {
                           "title": hit["_source"]["title"],
                           "score": hit["_score"],
                           "author": hit["_source"]["author"],
                           "link": hit["_source"]["link"],
                           "extract": utils.get_key_location(extractor, hit["_source"]["text"], list(keyphrases), hit["_source"]["title"])}
            output_list.append(output_json)

    
    return output_list, count


def text_search(es_client, index_name, query_text, thresh=1.6, top_n=10):

    count = es_client.cat.count(index=index_name, params={"format": "json"})[0]["count"]
    print(f'Showing recommendations from {count} documents ...')
    if not es_client.indices.exists(index=index_name):
        return "No records found"

    query2 = {
        "query": {
            "bool": {
                "must": {
                    "match": {
                        "title": query_text
                    }
                }
            }
        }
    }
    result = es_client.search(index=index_name, body=query2)
    total_match = len(result['hits']['hits'])
    print("Total Matches: ", str(total_match))

    print()
    counter = 0
    output_list = []
    if total_match > 0:
        for hit in result['hits']['hits']:
            counter += 1
            output_json = {"id": "12",
                           "title": hit["_source"]["title"],
                           "score": hit["_score"]}
            output_list.append(output_json)

    return output_list, count   


  
# @functions_framework.http
@app.route('/', methods=['POST'])
def search_fn():
    print(">>>>>> coming in flask")
    print(request)
    if request.method == "OPTIONS":
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }

        return ("", 204, headers)
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    ####  reading request #### 
    request_json = request.get_json(silent=True)

    if request_json["type"] == "audio":
        query_text = get_transcript(request_json["request_body"])
    else:
        query_text = request_json["request_body"]
    

    

    # semantic search
    semantic_output, count = semantic_search(client, index_name, query_text, top_n=5)

    # text search
    text_output, count = text_search(client, index_name, query_text)

    final_output = {
        "type": request_json["type"],
        "query": query_text,
        "semantic_search_results": semantic_output,
        "text_search_results": text_output,
        "total_count": count
    }
    headers = {"Access-Control-Allow-Origin": "*"}

    return (final_output, 200, headers)

if __name__ == '__main__':
    app.run(debug=True)