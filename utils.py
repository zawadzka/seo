from typing import List, Any
import re
import pandas as pd
import numpy as np
import streamlit as st
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import sentence_transformers
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

from google.cloud import bigquery
from google.oauth2 import service_account

service_account_file = f'{files_path}/play-project-328009-019f6ddaa96b.json'

m_handle = 'sentence-transformers/all-MiniLM-L6-v2'
bq_table = 'seo-project-392909.seo_dataset.data'
project_id = 'seo-project-392909'
secrets = st.secrets["gcp_service_account"]
bq_credentials = service_account.Credentials.from_service_account_file(secrets)
client = bigquery.Client(project=project_id, credentials=bq_credentials)

"""
Universal keyword set:
For the couponfollow.com site the most important (conversion) keywords 
are based directly on products - landing pages with coupons 
for particular shop/service. 
Those landing pages are located in the /site/ folder in the main domain. 
Possible keywords for those pages doesn't vary to the large extend - 
the main difference is the product name. 
"""
try:
    with open('universal_keyword_set.txt') as f:
        uks = f.readlines()
        uks = set(uks)
except FileNotFoundError as err:
    print(f'No keyword set loaded {err}')
    uks = {}


class InputData:
    """
    The InputData object contains inputs from Streamlit & BQ
    :param text: Text of description written by editors, scraped and saved in BQ
    :type text: str
    :param pr: Page rank value, calculated based on scraped internal links
    :type pr: float between 0 and 1 with step 0.01
    :param size: file size, from scraping
    :type size: int, in KB
    """
    keywordSet: list[Any]

    def __init__(self, content: str, name: str, pr: float, size: int,
                 time: float, sim_num: float = None,
                 content_length: int = None,
                 universal_keyword_set: set = uks):
        self.content = content
        self.name = name
        self._pr = pr
        self.size = size
        self.sim_sum = sim_num
        self.keywordSet = [re.sub(r'<plc>', self.name, k) for k in universal_keyword_set]
        self.time = time
        self.content_length = content_length

    @property
    def pr(self):
        return self._pr

    @pr.setter
    def pr(self, value):
        if value not in np.arange(0, 1, .01):
            raise ValueError("Page rank should be between 0 and 1")
        else:
            self._pr = value

    def similarity(self):
        """
        It calculates cosine similarity using huggingface bert model
        and tensorflow function -  equivalent of SentenceTransformer
        :return: sim_sum: float
        """


class BigQueryImport(InputData):
    def __init__(self, bq_table_name: str = bq_table) -> object:
        super().__init__(self)

    def bq_data_import(self, bq_table_name: str = bq_table):
        s = """
            SELECT time, sim_sum, full_content, size, pr, content_length
            FROM {}
        """
        sql = s.format(bq_table_name)
        page = client.query(sql)
        for row in page:
            self.time = page.time
            self.sim_sum = page.sim_sum
            self.size = page.size
            self.pr = page.pr
            self.content = page.full_content
            self.content_length = page.content_length




class SentenceTransformerTF:
    def __init__(self, sentence: str, model_handle: str = m_handle):
        self.tokenizer = AutoTokenizer.from_pretrained(model_handle)
        self.model = TFAutoModel.from_pretrained(model_handle)
        self.sentence = sentence
        self.embeddings = self.get_embeddings()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = tf.cast(tf.broadcast_to(tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)),
                                      tf.float32)

        sum_embeddings = tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = tf.clip_by_value(tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max)
        return sum_embeddings / sum_mask

    def get_embeddings(self) -> tf.keras.layers.Layer:
        inp = {'input_ids': self.to_chunks(self.sentence)}
        inp['token_type_ids'] = tf.ones_like(inp['input_ids'])
        inp['attention_mask'] = tf.where(inp['input_ids'] > 0, 1, 0)
        model_output = self.model(inp)
        embedding = self.mean_pooling(model_output, inp['attention_mask'])
        return embedding

    # function to navigate through the long tensor and slice it into parts with overlap

    @staticmethod
    def moving_window(a: tf.Tensor, overlap: int, window_size: int):
        """
        :param window_size: length of the shape[0] in resulting tensor
        :param overlap: int length of overlapping 
        :type a: tf.Tensor to be divided
        """
        a = tnp.reshape(a, (a.shape[1],))
        if a.shape[0] > 0:

            rep = 1 + (a.shape[0] - window_size) // (window_size - overlap)

            slicing = (
                    np.expand_dims(np.arange(window_size), 0) +
                    (window_size - overlap) * np.expand_dims(np.arange(rep), 0).T
            )
            max_val = (rep - 1) * (window_size - overlap) + window_size

            remaining = tnp.hstack(
                (a[max_val:], tnp.zeros(window_size - (a.shape[0] - window_size) % (window_size - overlap))))
            result = tnp.vstack((a[slicing], remaining))

            result = tf.cast(result, dtype=tf.int32)
            col = tnp.ones((result.shape[0], 1), dtype=np.int32)
            # adding start and end tokens
            # 101 - BERT's CLS (beginning) token, 102 - SEP (end) token
            result = tnp.append(101 * col, result, axis=1)
            result = tnp.append(result, 102 * col, axis=1)
            return result
        else:
            result = tf.zeros(window_size)
            result = tf.cast(result, dtype=tf.int32)
            result = tnp.append(tf.constant([101]), result)
            result = tnp.append(result, tf.constant([102]))
            return result

    # function divides content into tokens chunks. At the beginning of every chunk
    # BERT start token is added and at the end the SEP token.
    # overlap says how many tokes will be repeated between chunks
    # window_size is set for 128-2, because two places for the CLS and SEP tokens are needed
    def to_chunks(self, overlap: int = 4, window_size: int = 126, max_len: int = 512):
        tokens = self.tokenizer(self.sentence, add_special_tokens=False,
                                truncation=False, return_tensors='tf')
        token_chunks = self.moving_window(tokens['input_ids'], overlap, window_size)
        return token_chunks
