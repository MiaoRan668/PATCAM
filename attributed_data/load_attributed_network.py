import networkx as nx
import math
from sklearn.utils import shuffle
import numpy as np
from collections import namedtuple
import json
from keras.preprocessing.text import text_to_word_sequence
from attributed_data.read_attributed_data import load_glove_embedding,create_json_data

dir= "attributed_data/HepTH"
embed_len = 300
embedding_dir = "attributed_data/glove/"
embedding_path = embedding_dir + "glove.6B.%dd.txt" % embed_len


with open(dir + "/vocab.json", 'r',encoding='utf-8-sig') as jsonfile:
    vocab_dict = json.load(jsonfile)

with open(dir + "/caption.json", 'r',encoding='utf-8-sig') as jsonfile:
    caption_dataset = json.load(jsonfile)
vocab_size = len(vocab_dict)
max_sent_len = caption_dataset['max_sent_len']
annotation_map = caption_dataset['annotation']
embeddings = load_glove_embedding(embedding_path, vocab_dict, embed_len)

