# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 07:33:16 2024

@author: satyr

# @TBB_chat_bot, name TBB chat bot
# Use this token to access the HTTP API: 6514117454:AAH6DgqlBGx55oX71Z05-vSzQWXxmNBconw
"""

from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

class Generative_ChatBot():

    def __init__(self, model_checkpoint: str, tokenizer_checkpoint: str, verbose):
        '''
        Bot makes conversation session and reply to you using generative LLM
            - model_checkpoint: generative LLM model
            - tokenizer_checkpoint: tokenizer for the model
            - generative_config: settings of generation process (see GenerationConfig in transformers)
            - verbose: used for debugging (print extra info after each reply)
        '''
        self.model_checkpoint = model_checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.verbose = verbose
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(tokenizer_checkpoint)
        self.current_a = 'This is default answer. No answer was provided by Bot. Contact bot administrator.'
        self.gen_config = GenerationConfig(penalty_alpha = 0.6, do_sample = True,
                              top_k=5, temperature = 0.2, repetition_penalty =2.0,
                              max_new_tokens = 38, pad_token_id = self.tokenizer.eos_token_id)
    def retriev_answer(self, question) -> str:
        '''
        The name of function is ,isleading, this is not retrieval method but generation method
        It was named in this way to minimize change in other parts of code
        '''
        if question.strip() == '':
            return "Write something. I can not answer empty message!"
        prompt = f'''<|system|>
{"You are an engineer"}
<|user|>
{question}
<|assistant|>'''
        inputs = self.tokenizer([prompt], return_tensors='pt')
        outpust = self.model.generate(**inputs, generation_config = self.gen_config)
        reply = self.tokenizer.decode(outpust[0], skip_special_tokens = True).split("\n<|assistant|>")[1].strip()       
        if len(reply) > 0:
            return reply
        else:
            return "Alas, generative model return empty result, try another question please"
 

            
    def qa_iter(self, question: str = '') -> str:
        '''
        Single iteration of question and answer. The function listen your question, 
        find appropriate answer
        '''
        self.current_a = self.retriev_answer(question)
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

