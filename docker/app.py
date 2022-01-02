from flask import Flask, jsonify
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

import nltk
import re
import json
import joblib
import pandas as pd
import gensim
import logging

app = Flask(__name__)


def flatten_dict(dd, separator='_', prefix=''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }


def data_to_dataframe(lines_data: list, switch = False):
    temp_lst = []
    for line in lines_data:
        if not switch:
            temp_dict = ast.literal_eval(line)
            flat_temp_dict = flatten_dict(temp_dict)
        else:
            temp_dict = json.loads(line)
            flat_temp_dict = flatten_dict(temp_dict)
        temp_lst.append(flat_temp_dict)
    df = pd.DataFrame.from_dict(temp_lst)
    return df


def get_data(filename: str):
    with open(filename, 'r') as fin:
        data = fin.readlines()
    return data


def preprocess(doc):
    # lower case
    doc = doc.lower()
    # tokenize into words
    words = word_tokenize(doc)
    # remove stop words
    words = [re.sub("[\'./-=+].+", "" ,word) for word in words if word not in stoplist]
    words = [stemmer.stem(word) for word in words if len(word) > 2]

    return words


def get_prediction(rev_id, prod_id):
    # filter data
    df = df_review[(df_review.reviewerID == rev_id) & (df_review.asin == prod_id)]
    # processing
    df['review'] = df['reviewText'] + ' ' + df['summary']
    
    text =  df['review'].values.item(0)
    text = preprocess(text)
    # tf idf using existing vocab
    test = VECTORIZER.infer_vector(text)
    # predict
    return MODEL.predict([test])



# loading data
data_review = get_data("data/tf_interview_review_FASHION.json")
df_review = data_to_dataframe(data_review, switch=True)


# loading models    
MODEL = joblib.load("models/model20220102.pkl")
VECTORIZER = gensim.models.Word2Vec.load("models/doc_embedding_20220102")
stemmer = PorterStemmer()
stoplist = stopwords.words('english') + list(punctuation)


@app.route('/predict/<reviewer_id>/<product_id>')
def predict(reviewer_id, product_id):
    '''
        This method takes in reviewer id and product id to predict the ratings
    '''
    try:
        if (not reviewer_id) and (not product_id):
            return 'incorrect url'
        else:
            return jsonify({"score": get_prediction(reviewer_id, product_id).item(0)})  
    except Exception as e:
        logging.debug(e)
        return "Try Checking your input"
        

# app.run(debug=True)