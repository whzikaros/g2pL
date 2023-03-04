import argparse
from time import monotonic
import opencc
import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from g2pL.model import G2pL
from transformers import BertConfig
from transformers.tokenization_bert import BertTokenizer
from .utils import *
from .vocab import *
from torch.utils.data import DataLoader,Dataset
from transformers import logging
 
logging.set_verbosity_error()

word_vocab_url="https://huggingface.co/Megumism/g2pL/resolve/main/tencent_vocab.txt"
model_url="https://huggingface.co/Megumism/g2pL/resolve/main/best_model.pt"
embedding_url="https://huggingface.co/Megumism/g2pL/resolve/main/saved_word_embedding_1000000.pkl"

dir=os.path.dirname(os.path.abspath(__file__))

file_dir=os.path.join(os.path.dirname(dir),"g2pL_files")

import collections

class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

class Trie:
    """
    In fact, this Trie is a letter tree.
    root is a fake node, its function is only the begin of a word, same as <bow>
    the the first layer is all the word's possible first letter, for example, '中国'
        its first letter is '中'
    the second the layer is all the word's possible second letter.
    and so on
    """
    def __init__(self, use_single=True):
        self.root = TrieNode()
        self.max_depth = 0
        if use_single:
            self.min_len = 0
        else:
            self.min_len = 1

    def insert(self, word):
        current = self.root
        deep = 0
        for letter in word:
            current = current.children[letter]
            deep += 1
        current.is_word = True
        if deep > self.max_depth:
            self.max_depth = deep

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)

            if current is None:
                return False
        return current.is_word

    def enumerateMatch(self, str, space=""):
        """
        Args:
            str: 需要匹配的词
        Return:
            返回匹配的词, 如果存在多字词，则会筛去单字词
        """
        matched = []
        while len(str) > self.min_len:
            if self.search(str):
                matched.insert(0, space.join(str[:])) # 短的词总是在最前面
            del str[-1]

        if len(matched) > 1 and len(matched[0]) == 1: # filter single character word
            matched = matched[1:]

        return matched


class TestDataset(Dataset):
    '''
    返回dataset类型，打包了所有返回值
    '''
    def __init__(self, sent_list, params):
        """
        Args:
            sent_list(list):要测试的句子们
            params(dict): 1.tokenizer
                          2.word_vocab
                          3.class2idx
                          4.lexicon_tree
                          5.max_seq_length
                          6.max_scan_num
                          7.max_word_num
                          8.index
                                                    
        """
        super(TestDataset, self).__init__()
        self.tokenizer = params['tokenizer'] 
        self.max_seq_length = params['max_seq_length']
        self.class2idx = params['class2idx']
        self.word_vocab = params['word_vocab']
        self.lexicon_tree = params['lexicon_tree']
        self.max_scan_num = params['max_scan_num']        
        self.max_word_num = params['max_word_num']
        self.poly_index = params['index']
        
        self.sents = sent_list
        self.num_classes = len(self.class2idx)
        self.total_size = len(self.sents)
        
    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        sent = self.sents[index].strip()     

        poly_idx=self.poly_index+1        

        inputs = self.tokenizer.encode_plus(
            sent, add_special_tokens=True, max_length=self.max_seq_length, return_token_type_ids=True, pad_to_max_length=True, verbose=False
        )         
        #print(inputs)
        input_ids=inputs["input_ids"] 

        token_type_ids=inputs["token_type_ids"]
        attention_mask=inputs["attention_mask"]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)        
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        poly_idx = torch.tensor(poly_idx, dtype=torch.long) #多音字在句子的位置
        text=list(sent)
        text.insert(0, '[CLS]')
        text.append('[SEP]')
            
        matched_word_ids = np.zeros((self.max_seq_length, self.max_word_num), dtype=np.int)
        matched_word_mask = np.zeros((self.max_seq_length, self.max_word_num), dtype=np.int)     
        matched_words, _ = sent_to_matched_words_boundaries(text, self.lexicon_tree, self.max_word_num)
        
        sent_length = len(text)
        for idy in range(sent_length):
            now_words = matched_words[idy]
            now_word_ids = self.word_vocab.convert_items_to_ids(now_words)
            matched_word_ids[idy][:len(now_word_ids)] = now_word_ids
            matched_word_mask[idy][:len(now_word_ids)] = 1  
                  
        input_matched_word_ids=torch.tensor(matched_word_ids, dtype=torch.long) 
        input_matched_word_mask=torch.tensor(matched_word_mask, dtype=torch.long)
        
        return (
            input_ids, 
            token_type_ids,
            attention_mask, 
            input_matched_word_ids,
            input_matched_word_mask,
            poly_idx
            )

class G2PL:
    def __init__(self):
        self.config = BertConfig.from_pretrained(os.path.join(dir,"bert_config/config_g2pL.json"))
        self.converter = opencc.OpenCC('t2s.json')
        self.poly_dic=os.path.join(dir,"polyphone_dic.json")
        self.pinyin_dic=os.path.join(dir,"pinyin_dic.json")
        self.tone_dic=os.path.join(dir,"tone2totone1.json")
        with open(self.poly_dic, "r", encoding="utf-8") as f:
            self.polyphones = json.load(f)
        with open(self.pinyin_dic, "r", encoding="utf-8") as f:
            self.monotonic_chars = json.load(f)
        with open(self.tone_dic, "r", encoding="utf-8") as f:
            self.tones = json.load(f)        
        self.tokenizer=BertTokenizer.from_pretrained(os.path.join(dir,"bert_config"))
                  
        with open(os.path.join(dir,"class2idx.pkl"), "rb") as f:
            self.class2idx = pickle.load(f)
        self.num_classes = len(self.class2idx)

        self.embedding_file=os.path.join(file_dir,"saved_word_embedding_1000000.pkl")
        #self.embedding_file="/media/data2/wanhongzhi/my_polyphone_inference/data/embedding/saved_word_embedding_1000000.pkl"
        if not os.path.exists(self.embedding_file):
            download_model(embedding_url,self.embedding_file)
            print("saved_word_embedding_1000000.pkl download successfully!") 
            time.sleep(5)   
                                        
        with open(self.embedding_file, 'rb') as f:
            self.pretrained_word_embedding = pickle.load(f)
        with open(os.path.join(dir,"matched_word.txt"),'r') as f:
            matched_words=f.readlines()
            
        self.word_vocab_file=os.path.join(file_dir,"tencent_vocab.txt")
        #self.word_vocab_file="/media/data2/wanhongzhi/my_polyphone_inference/data/vocab/tencent_vocab.txt"
        if not os.path.exists(self.word_vocab_file):
            download_model(word_vocab_url,self.word_vocab_file)
            print("tencent_vocab.txt download successfully!") 
            time.sleep(5) 
                                
        self.lexicon_tree=build_lexicon_tree_from_vocabs([self.word_vocab_file], scan_nums=[1000000]) 
    
        self.word_vocab = ItemVocabArray(items_array=matched_words, is_word=True, has_default=False, unk_num=5)   
         
        self.inference_model=os.path.join(file_dir,"best_model.pt")
        #self.inference_model="/media/data2/wanhongzhi/my_polyphone_inference/g2pL_files/best_model.pt"
        if not os.path.exists(self.inference_model):
            download_model(model_url,self.inference_model)  
            print("best_model.pt download successfully!")  
            time.sleep(5)
                      
        self.model = G2pL.from_pretrained( #和原始bert核心区别在BertLayer类里
            self.inference_model, config=self.config,
            pretrained_embeddings=self.pretrained_word_embedding,
            num_labels=self.num_classes,
            seq_len=128
        )

    def predict(self, sent, index, model):
        if isinstance(sent,str):
            sent_list=[]
            sent_list.append(sent)
        elif isinstance(sent,list):
            sent_list=sent
        params={
            'tokenizer': self.tokenizer,
            'word_vocab': self.word_vocab,
            'class2idx': self.class2idx,
            'lexicon_tree': self.lexicon_tree,
            'max_seq_length': 128,
            'max_scan_num': 1000000,
            'max_word_num': 3, #5
            'index' : index,
        }
        self.dataset=TestDataset(sent_list, params)
        dataloader = DataLoader(self.dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0)
        all_preds = []
        all_poly_ids=[] 
        all_input_ids=[]
        for batch in dataloader:
            input_ids, token_type_ids, attention_mask, input_matched_word_ids, input_matched_word_mask, poly_ids = batch
          
            inputs = {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "matched_word_ids": input_matched_word_ids,
                "matched_word_mask": input_matched_word_mask,
                "poly_ids": poly_ids,
                }

            model.eval()
            with torch.no_grad():
                logits = model(**inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            input_ids=input_ids.cpu().numpy().tolist()
            all_input_ids+=input_ids
            all_poly_ids.append(poly_ids.cpu().numpy())
            all_preds.append(preds)
        polys = np.concatenate(all_poly_ids, axis=0)
        preds = np.concatenate(all_preds, axis=0)
        if len(preds)==1:
            out_pinyin=list(self.class2idx.keys())[list(self.class2idx.values()).index(preds[0])]
        return out_pinyin
                    
    def __call__(self, sent, tone=2):
        """
        tone表示声调，不同的值代表不同声调风格的拼音：
            tone为0表示普通风格，不带声调。。如：ni hao
            tone为1表示标准声调风格，拼音声调在韵母第一个字母上。如：nǐ hǎo
            tone为2表示用数字代表声调并且放在每个拼音最后。如：ni3 hao3
        """
        sent=self.converter.convert(sent) #繁体转简体
        result=[]              
        for idx, char in enumerate(sent):
            if char in self.polyphones.keys():
                char_pinyin=self.predict(sent, idx, self.model)
                if tone==0:
                    char_pinyin=char_pinyin[:-1]
                elif tone==1:
                    char_pinyin=self.tones[char_pinyin]
                result.append(char_pinyin)
            elif char in self.monotonic_chars.keys():
                char_pinyins=self.monotonic_chars[char]
                char_pinyin=char_pinyins.split(",")[0]
                if tone==0:
                    char_pinyin=char_pinyin[:-1]
                elif tone==1:
                    char_pinyin=self.tones[char_pinyin]
                result.append(char_pinyin)
            else:
                result.append(char)
        return result        
                            
def download_model(url,model_path):
    r = requests.get(url)
    with open(model_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                