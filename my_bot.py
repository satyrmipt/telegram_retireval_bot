# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 07:33:16 2024

@author: satyr

# @TBB_chat_bot, name TBB chat bot
# Use this token to access the HTTP API: 6514117454:AAH6DgqlBGx55oX71Z05-vSzQWXxmNBconw
"""

import pandas as pd
import numpy as np
from my_tokenize_vectorize import Tokenizer, Glove_vectorizer, SentBERT_vectorizer, Reranker
import faiss
from sklearn.preprocessing import normalize
from pprint import pprint



class ChatBot():

    def __init__(self, vector_db_path: str, context_len: int =3, 
                 similarity_type: str = 'retrieval', vectorization_type: str = 'glove', character: str = 'Leonard', 
                 rerank_param: tuple = (None, None, None),
                 answer_max_len: int = 4096, verbose: int =0 ):
        '''
        Bot makes conversation session and reply to you.
        This construcor reads heavy vectorized file/database to look for suitable answers during conversation so it take a wile
        Inputs:
            - vector_db_path:path to file wich will be used to retriev answers and context
            - contexlen: max number of previous question and answers (both counts) to take into account during retrieval preocess (depth of context),  must be in [0,5]
            - similarity_type: method to find appropriate answer. Only 'retrieval' implemented right now
            - vectorization_type: method to get text embeddigs:
                 - 'glove' for Glove embeddings
                 - 'sent_encode' for SBERT
            - rerank_param: default (None, None, None), ('cross_enc', top_n_to_rerank, path_to_cross_enc_model)
            - character: character of bot. Do not change if you don't participate in data gathering
            - answer_max_len: any reply will be truncated to this number of characters, usefull for Telegram since it hase this restriction
            - verbose: used for debugging (print extra info after each reply)
        '''
        self.context_len = context_len
        self.prev_conversation = []
        self.qa_iter_count = 0 
        self.current_q = ''
        self.current_a = ''
        self.character = character.lower()
        self.session_stop_word = 'stop'
        self.debug_verbose = verbose
        self.similarity_type = similarity_type
        self.answer_max_len = answer_max_len
        self.verbose = verbose
        self.rereank_type, self.rerank_res_cnt, self.path_to_model = rerank_param
        self.rerank_model = None
        if similarity_type != 'retrieval':
            raise ValueError(f'{similarity_type=} not implemented yet, see documntation')

        if context_len < 0 or context_len > 5:
            raise ValueError(f'{context_len=} not in [1...5] range. Dataset for retrieval algorithm was build with maximum 5 previous interactions')
        # self.vdf = pd.read_excel('C:\\Users\\satyr\\Documents\\edu\\nlp2\\hw1\\data\\eng_script_vectorized.xlsx')
        if self.verbose !=0: print('Reading character conversation vector db, wait a bit...')
        self.vdf = pd.read_pickle(vector_db_path)
        if self.character not in self.vdf['person'].str.lower().unique():
            raise ValueError(f"{self.character=} is absent is training data. Use one of {self.vdf['person'].str.lower().unique()}")
        else:
            # reset index for future search by dense index:
            self.vdf = self.vdf[self.vdf['person'].str.lower() == self.character].reset_index(drop=True)
            self.vdf = self.vdf.reset_index(drop=True)
            if self.verbose !=0: print(f"Working dataset shape: {self.vdf.shape}")
        
        if self.similarity_type == 'retrieval':
            if vectorization_type == 'glove':
                # create tokenizer and vectorizer
                self.tk = Tokenizer(tokenizer = 'wordpunct_tokenize', lower=True, lang = 'english' )
                if self.verbose !=0: print('Reading vector db to vectorize user responses , wait a bit...') 
                self.v =  Glove_vectorizer(dim = 50, path_to_emb_folder = "data\\")
            elif vectorization_type == 'sent_encode':
                self.tk = Tokenizer(tokenizer = 'dummy', lower=True, lang = 'english' )
                if self.verbose !=0: print('Reading SBERT model, wait a bit...') 
                self.v = SentBERT_vectorizer(checkpoint="all-mpnet-base-v2")
            else:
                raise ValueError(f'{vectorization_type=} not implemented for {similarity_type=}, see documntation')
            # create FAISS index list. It requires to normalize vectros before comparison
            self.faiss_ind_list = []
            if self.verbose !=0: print('Creating FAISS indexes, wait a bit...') 
            for i in range(1, self.context_len+1):
                curr_emb = np.stack(self.vdf[f"{self.v.col_prefix}v_text_{i}_shift"].to_list(), axis=0)
                curr_emb_norm = normalize(curr_emb, axis=1, norm='l2')
                curr_ind = faiss.index_factory(self.v.dim, "Flat" , faiss.METRIC_INNER_PRODUCT) # cosine similarity
                curr_ind.train(curr_emb_norm)
                curr_ind.add(curr_emb_norm)
                self.faiss_ind_list.append(curr_ind)
            if self.verbose !=0: print(f'Count of FAISS indexes {len(self.faiss_ind_list) = } must be equal {self.context_len = }') 
            
            if self.rereank_type == "cross_enc":
                if self.verbose !=0: print('Reading CrossEncoder model, wait a bit...')
                self.rerank_model = Reranker(checkpoint = 'model\\crerankingeval-30e-4000-ms-marco-MiniLM-L-6-v2')

    def retriev_answer(self) -> str:
        '''Calculate answer in case of similarity_type='retrieval' '''
        if self.similarity_type != 'retrieval':
            raise ValueError(f'{self.similarity_type=} but retrieve_answer was called')

        # Calculate vector of current question and previous context
        # We use weigted average: the latest parts of dialog weigts more
        # then oldest with. Weight is 1/n**2 where n=1 current question and
        # n=5 for speach wich wash 5 iterations before it
        vect_conv_list = np.array(self.v.vectorize_tokens(self.tk.tokenize_corpus(self.prev_conversation)))
        # length on non-zero prev conversations
        nonzero_vcl = np.array([v for v in vect_conv_list if not (v==self.v.zero_vect).all()])
        vcl_len = len(nonzero_vcl)
        print(f"{nonzero_vcl.shape=}, {vcl_len=}")
        if vcl_len > 0:
            weights = np.array(list(reversed([1/(i+1)**2 for i in range(vcl_len)])))
            #weights = np.array([1 for i in range(vcl_len)])
            vect_conv_weighted = np.average(a = nonzero_vcl, weights = weights, axis = 0)
            print(f"{vect_conv_weighted.shape=}")
            vect_conv_wn = normalize(vect_conv_weighted.reshape(1, -1), axis=1, norm='l2')
            print(f"{vect_conv_wn.shape=}")
            print(f"{vect_conv_wn=}")
        else:
            return "I don't understand you, try again please"
        if self.rereank_type is None:
            # Find first nearest in FAISS 
            fais_ind = self.faiss_ind_list[len(self.prev_conversation)-1]
            dist, ind = fais_ind.search(x = vect_conv_wn.astype(np.float32), k=1)
            return self.vdf.loc[ind[0][0], 'text']
        elif self.rereank_type == "cross_enc":
            # Find self.rerank_res nearest in FAISS:
            fais_ind = self.faiss_ind_list[len(self.prev_conversation)-1]
            dist, ind = fais_ind.search(x = vect_conv_wn.astype(np.float32), k=self.rerank_res_cnt)
            # index list already sorted by dist (cosine similarity)
            result_list = [self.vdf.loc[i, 'text'] for i in ind[0]]
            print("------1st step results-------", *result_list, sep='\n\t')
            rerank_scores = self.rerank_model.rerank_results([[self.current_q, res] for res in result_list ])
            reranked_result_list = [x for _, x in sorted(zip(rerank_scores, result_list), key=lambda pair: pair[0], reverse=True)]
            print("------2nd step results-------", *reranked_result_list, sep='\n\t')
            return reranked_result_list[0]
        else:
            raise ValueError(f'{self.rereank_type=} not implemented yet, see available types in doc')

    
    def generate_answer(self) -> str:
        pass
            
    def qa_iter(self, question: str = '') -> str:
        '''
        Single iteration of question and answer. The function listen your question, 
        find appropriate answer, log both for future interactions
        '''
        #self.current_q = question.lower()
        self.current_q = question
        if len(self.prev_conversation) >= self.context_len:
            self.prev_conversation.pop(0)
        self.prev_conversation.append(self.current_q)
        if self.similarity_type == 'retrieval':
            self.current_a = self.retriev_answer()
        self.current_a = self.current_a[:self.answer_max_len] # telegram has message length limit
        if len(self.prev_conversation) >= self.context_len:
            self.prev_conversation.pop(0)
        self.prev_conversation.append(self.current_a)
        self.qa_iter_count += 1
        return self.current_a

    def qa_session(self):
        '''
        this function are used only for debugging in local env, use Telegram Bot wrapper in prod
        '''
        if self.qa_iter_count == 0:
            print(f"Hey, I'm {self.character} from TBB. Print '{self.session_stop_word}' to reset session.\n")
        while True:
            self.current_q = input().lower()
            if self.current_q == 'stop':
                print("Good bye.")
                break
            self.qa_iter(self.current_q)
            print(f"You said: '{self.current_q}'. Answer is {self.current_a}\n")
            if self.debug_verbose !=0:
                print(self.__dict__)



# bot = ChatBot(vector_db_path = "C:\\Users\\satyr\\Documents\\edu\\nlp2\\hw1\\data\\eng_script_vectorized.xlsx",
#               context_len = 3, 
#               similarity_type = 'retrieval', 
#               character = 'Leonard',
#               verbose=1 )
# bot.qa_session()

