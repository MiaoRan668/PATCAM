import numpy as np
from model.attentionlayer import Attention, AverageAttention
from keras.constraints import max_norm
import keras.backend as K
from attributed_data.read_attributed_data import read_attribute_graph
from keras.models import Sequential, Model
from keras.layers import GRU, Embedding, concatenate, Input, Dense, Lambda, Masking, Add, add,average
from attributed_data.load_attributed_network import *
from model.dynet import DynamicNet
import pandas as pd
from attributed_data.evaluation import test_linkpredict,classification
from sklearn.model_selection import train_test_split
import config



def make_dict():
    idx = list(node_mapping_dic.values())
    stat_dict = None
    if model.state_emb:
        stat_dict = dict(zip(node_mapping_dic.keys(), model.state_emb.get_weights()[0][idx]))
    emb_dict = dict(zip(node_mapping_dic.keys(), model.get_features(idx, feat_mat=txt_array, batch_size=predict_batch_sz)))
    return emb_dict, stat_dict



def evaluation():
    test_linkpredict(test_path, model.embed_score_model, embedding_dict, state_dict,
                        directed=True, batch_size=predict_batch_sz)



directory = "attributed_data/HepTH"
directed = True
weighted = False

path = directory+'/hep_train_0.1.edgelist'
test_path = directory + '/hep_test_negative_0.1.txt'



nx_G = read_attribute_graph(path,directed=directed, weighted=weighted)

node_set = set(nx_G.nodes)
node_size = len(node_set)

print(len(nx_G.edges))
print(node_size)

node_text_set = {}
node_mapping_dic = {}
node_inv_map_dic = {}

txt_array = np.zeros([node_size, max_sent_len]) #29896,35

with open(directory + "/vocab.json", 'r',encoding='utf-8-sig') as jsonfile:
    vocab_dict = json.load(jsonfile)

with open(directory + "/caption.json", 'r',encoding='utf-8-sig') as jsonfile:
    caption_dataset = json.load(jsonfile)
vocab_size = len(vocab_dict)
max_sent_len = caption_dataset['max_sent_len']
annotation_map = caption_dataset['annotation']

for i, id in enumerate(node_set):
    words = annotation_map[str(id)] #读取caption.json中的文件
    node_text_array = np.zeros([1, max_sent_len],dtype=np.int32) # "max_sent_len": 35
    if len(words) > 0:
        node_text_array[0,-len(words):] = [vocab_dict[w] for w in words]
        node_text_set[id] = node_text_array
    node_mapping_dic[id] = i
    node_inv_map_dic[i] = id
    txt_array[i]= node_text_array   #记录节点中的文本



sent_input = Input((max_sent_len,), name='sent_input') #shape=(?,35)
word_embed = Embedding(vocab_size + 1, embed_len, mask_zero=True, name='word_embedding',
                       weights=[embeddings], trainable=False)(sent_input)#shape=(?,35,300)


sent_embed = AverageAttention(alpha=4, keepdims=True)(word_embed)
# txtmodel.add(l2_norm)

node_input = Input((1,), name='node_input')
node_embed = Embedding(node_size, embed_len, name='node_embed')(node_input)

attributed_embed = add([sent_embed, node_embed])    # 包括句子的embedding和结构embedding

attributed_embed = Lambda(lambda x : K.l2_normalize(x, axis=-1))(attributed_embed)

attributed_model = Model(inputs=[node_input,sent_input],outputs=attributed_embed)

#——————————train——————————————————————
model = DynamicNet(attributed_model, node_size, state_dim=10, directed=directed,
                    run_dynamics=False)

edges = list(nx_G.edges())
nb_edges = len(edges)
from_node = np.zeros([nb_edges], dtype=np.int32)
to_node = np.zeros([nb_edges], dtype=np.int32)

print(nb_edges)
for i,e in enumerate(edges):
    from_node[i]=node_mapping_dic[edges[i][0]]
    to_node[i] = node_mapping_dic[edges[i][1]]




dropout_ratio = config.dropout_ratio
predict_batch_sz=config.predict_batch_sz
batch_sz = config.batch_sz
margin_aspect = config.margin_aspect
margin_overall = config.margin_overall

update_net_cfg_per_n_iter = config.update_net_cfg_per_n_iter
max_iter= config.max_iter

# 控制是否输入文本信息
#txt_array=np.zeros(shape=txt_array.shape) 
for it in range(max_iter):
    if it % update_net_cfg_per_n_iter == 0:

        net_corrupt = model.dropout_network(from_node, to_node, values=None, ratio=dropout_ratio)
        print('Reconfig network')

    print('Iteration %d/%d'%(it+1, max_iter))
    ix = np.random.permutation(len(from_node))
    neg_spl_nodes = np.random.randint(0, node_size, size=len(from_node), dtype=np.int32)
    model.fit((from_node[ix], to_node[ix], neg_spl_nodes), txt_array, edges=net_corrupt,
              margin_con=margin_aspect, margin_overall=margin_overall,
              batch_size=batch_sz, pred_batch_size=predict_batch_sz)

    # if (it + 1) % 10 == 0:
    #     evaluation()

post_train_it = 20
net_corrupt = model.dropout_network(from_node, to_node, values=None, ratio=0)
for it in range(post_train_it):
    print('Post Iteration %d/%d' % (it + 1, post_train_it))
    ix = np.random.permutation(len(from_node))
    neg_spl_nodes = np.random.randint(0, node_size, size=len(from_node), dtype=np.int32)
    model.fit((from_node[ix], to_node[ix], neg_spl_nodes), txt_array, edges=net_corrupt,
              margin_con=margin_aspect, margin_overall=margin_overall,
              batch_size=batch_sz, pred_batch_size=predict_batch_sz)



embedding_dict, state_dict = make_dict()
embed_save_path = directory + '/embed'
np.save(embed_save_path, np.hstack([np.expand_dims(list(embedding_dict.keys()), axis=1), list(embedding_dict.values())]))

evaluation()
