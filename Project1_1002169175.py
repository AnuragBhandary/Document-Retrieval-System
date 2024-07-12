import os
import math
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# READ DOCUMENTS
corpusroot = './US_Inaugural_Addresses'
documents = {}
for filename in os.listdir(corpusroot):
    if filename.endswith('.txt'):
        with open(os.path.join(corpusroot, filename), 'r') as file:
            text = file.read().lower()
            tokens = tokenizer.tokenize(text)
            tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
            documents[filename] = tokens

# CALCULATING DF AND TF
df = defaultdict(int)
tf = defaultdict(lambda: defaultdict(int))
for filename, tokens in documents.items():
    unique_tokens = set(tokens)
    for token in unique_tokens:
        df[token] += 1
    for token in tokens:
        tf[filename][token] += 1

N = len(documents)  # Total number of documents

# MAKING A DICTIONARY OF IDF VALUES
idf = {token: math.log(N / df[token], 10) for token in df}

# GETTING IDF
def getidf(token):
    token = stemmer.stem(token.lower())
    return idf.get(token, -1)

# GETTING THE TF-IDF WEIGHT
def getweight(filename, token):
    token = stemmer.stem(token.lower())
    if token not in tf[filename]:
        return 0
    tf_weight = 1 + math.log(tf[filename][token], 10)
    idf_weight = getidf(token)
    return tf_weight * idf_weight

# COS SIMILARITY
def cos_similarity(vec1, vec2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    norm_vec1 = math.sqrt(sum(v1 * v1 for v1 in vec1))
    norm_vec2 = math.sqrt(sum(v2 * v2 for v2 in vec2))
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)

# TO PROCESS THE QUERY
def query(qstring):
    query_tokens = tokenizer.tokenize(qstring.lower())
    query_tokens = [stemmer.stem(token) for token in query_tokens if token not in stop_words]
    
    query_tf = defaultdict(int)
    for token in query_tokens:
        query_tf[token] += 1
    
    # CONSTRUCTING QUERY VECTOR
    query_vector = [(1 + math.log(query_tf[token], 10)) for token in query_tokens]
    
    # NORMALIZING QUERY VECTOR
    query_norm = math.sqrt(sum(val ** 2 for val in query_vector))
    if query_norm != 0:
        query_vector = [val / query_norm for val in query_vector]
    
    max_score = 0
    best_document = None
    for filename in documents:
        doc_vector = [getweight(filename, token) for token in query_tokens]
        
        # NORMALIZING DOCUMENT VECTOR
        doc_norm = math.sqrt(sum(val ** 2 for val in doc_vector))
        if doc_norm != 0:
            doc_vector = [val / doc_norm for val in doc_vector]
        
        score = cos_similarity(query_vector, doc_vector)
        if score > max_score:
            max_score = score
            best_document = filename
    
    return best_document, max_score


print("%.12f" % getidf('democracy'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('states'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
print("%.12f" % getweight('23_hayes_1877.txt','public'))
print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
print("%.12f" % getweight('37_roosevelt_franklin_1933.txt','leadership'))
print("--------------")
print("(%s, %.12f)" % query("states laws"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("world civilization"))
