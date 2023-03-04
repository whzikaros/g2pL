import time
import os
import json
from tqdm import tqdm, trange
import numpy as np
import pickle
from .preprocess.lexicon_tree import Trie


def build_lexicon_tree_from_vocabs(vocab_files, scan_nums=None):
    """
    根据词汇表建立词汇树。
    Args:
        vocab_files(list): 字符形式的词表路径，外面套一层[]，此处为tencent_vocab.txt,有8824350个词。
        scan_nums(list): 最大浏览词表树中多少词，外面套一层[]，此处为1000000。
    Return:
        返回词典树，Trie类的实例。
    """
    # 1.获取词汇表
    #print(vocab_files)
    vocabs = set()
    if scan_nums is None:
        length = len(vocab_files)
        scan_nums = [-1] * length

    for file, need_num in zip(vocab_files, scan_nums):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_line_num = len(lines) #词汇表中字符的数量
            if need_num >= 0: # need_num = scan_nums
                total_line_num = min(total_line_num, need_num)

            # line_iter = trange(total_line_num) //去掉建立树的进度条
            # for idx in line_iter:
            for idx in range(total_line_num):
                line = lines[idx]
                line = line.strip()
                items = line.split()
                word = items[0].strip()
                vocabs.add(word)
    vocabs = list(vocabs)
    vocabs = sorted(vocabs)
    # 2.建立词典树
    lexicon_tree = Trie()
    for word in vocabs:
        lexicon_tree.insert(word)
    return lexicon_tree

def get_corpus_matched_word_from_lexicon_tree(files, lexicon_tree):
    """
    数据类型统一为json格式, {'text': , 'label': }
    Args:
        files: corpus data files
        lexicon_tree: built lexicon tree

    Return:
        total_matched_words: all found matched words
    """
    # 先将数据集每个字符隔开作为列表元素形成列表。
    total_matched_words = set()
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                line_list = list(line)

                sent_matched_words = sent_to_matched_words_set(line_list, lexicon_tree)
                _ = [total_matched_words.add(word) for word in sent_matched_words]

    total_matched_words = list(total_matched_words)
    total_matched_words = sorted(total_matched_words)
    with open("matched_word.txt", "w", encoding="utf-8") as f:
        for word in total_matched_words:
            f.write("%s\n"%(word))

    return total_matched_words

def sent_to_matched_words_set(sent, lexicon_tree, max_word_num=None):
    """return matched words set
        根据词典树和句子返回匹配到的词集。
    """
    sent_length = len(sent)
    sent_words = [[] for _ in range(sent_length)]
    matched_words_set = set()
    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]  # speed using max depth
        words = lexicon_tree.enumerateMatch(sub_sent)

        _ = [matched_words_set.add(word) for word in words]
    matched_words_set = list(matched_words_set)
    matched_words_set = sorted(matched_words_set)
    return matched_words_set

def build_pretrained_embedding_for_corpus(
        embedding_path,
        word_vocab,
        embed_dim=200,
        max_scan_num=1000000,
        saved_corpus_embedding_dir=None,
        add_seg_vocab=False
):
    """
    Args:
        embedding_path: 预训练的word embedding路径
        word_vocab: corpus的word vocab，根据从词表树中匹配到的词制作的词表类
        embed_dim: 维度
        max_scan_num: 最大浏览多大数量的词表
        saved_corpus_embedding_dir: 这个corpus对应的embedding保存路径
    Return:
        pretrained_emb：
            numpy形式返回的二维数组，形状为(232896, 200)。
            第一维表示词的个数，第二维表示词embedding维度。

    """
    saved_corpus_embedding_file = os.path.join(saved_corpus_embedding_dir, 'saved_word_embedding_{}.pkl'.format(max_scan_num))

    if os.path.exists(saved_corpus_embedding_file): #此处只有word_embedding.txt
        with open(saved_corpus_embedding_file, 'rb') as f:
            pretrained_emb = pickle.load(f)
        return pretrained_emb, embed_dim

    embed_dict = dict()
    if embedding_path is not None:
        embed_dict, embed_dim = load_pretrain_embed(embedding_path, max_scan_num=max_scan_num, add_seg_vocab=add_seg_vocab)
    scale = np.sqrt(3.0 / embed_dim) # 平方根,作为随机采样的上下限。
    pretrained_emb = np.empty([word_vocab.item_size, embed_dim])

    matched = 0
    not_matched = 0

    for idx, word in enumerate(word_vocab.idx2item): # 遍历从词典树中获取的1000000(来源是数据集),在的就直接用,不在的则随机采样
        if word in embed_dict: 
            pretrained_emb[idx, :] = embed_dict[word]
            matched += 1
        else:
            pretrained_emb[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
            not_matched += 1

    pretrained_size = len(embed_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, oov:%s, oov%%:%s" % (
    pretrained_size, matched, not_matched, (not_matched + 0.) / word_vocab.item_size))

    with open(saved_corpus_embedding_file, 'wb') as f:
        pickle.dump(pretrained_emb, f, protocol=4)

    return pretrained_emb, embed_dim

def load_pretrain_embed(embedding_path, max_scan_num=1000000, add_seg_vocab=False):
    """
    从pretrained word embedding(word_embedding.txt)中读取前max_scan_num的词向量
    制作token到word embedding映射的字典,有max_scan_num个token。
    Args:
        embedding_path(str): 词向量路径,此处为word_embedding.txt
        max_scan_num(int): 最多读多少,和从词典树读取的词数一样
    Return:
        embed_dict(dict): 一个token到word_embedding映射的字典,{token: word_embedding},有max_scan_num个元素
        embed_dim(int): word_embedding的维度,用的预训练embedding为200。
    """

    ## 如果是使用add_seg_vocab, 则全局遍历
    if add_seg_vocab:
        max_scan_num = -1

    embed_dict = dict()
    embed_dim = -1
    with open(embedding_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if max_scan_num == -1:
            max_scan_num = len(lines)
        max_scan_num = min(max_scan_num, len(lines))
        line_iter = trange(max_scan_num)
        for idx in line_iter:
            line = lines[idx]
            line = line.strip()
            items = line.split()
            if len(items) == 2: # word_embedding.txt文件第一行['8824330', '200']
                embed_dim = int(items[1])
                continue
            elif len(items) == 201: #词本体+维度200，一共201个元素
                token = items[0]
                embedd = np.empty([1, embed_dim])
                embedd[:] = items[1:]
                embed_dict[token] = embedd
            elif len(items) > 201:
                print("++++longer than 201+++++, line is: %s\n" % (line))
                token = items[0:-200]
                token = "".join(token)
                embedd = np.empty([1, embed_dim])
                embedd[:] = items[-200:]
                embed_dict[token] = embedd
            else:
                print("-------error word-------, line is: %s\n"%(line))

    return embed_dict, embed_dim

def sent_to_matched_words_boundaries(sent, lexicon_tree, max_word_num=None):
    """
    输入一个句子和词典树, 返回句子中每个字所属的匹配词, 以及该字的词边界(边界分为8种)
    字可能属于以下几种边界:
        B-: 词的开始, 0
        M-: 词的中间, 1
        E-: 词的结尾, 2
        S-: 单字词, 3
        BM-: 既是某个词的开始, 又是某个词中间, 4
        BE-: 既是某个词开始，又是某个词结尾, 5
        ME-: 既是某个词的中间，又是某个词结尾, 6
        BME-: 词的开始、词的中间和词的结尾, 7

    Args:
        sent: 输入的句子, 一个字的数组, 如：['[CLS]', '华', '盛', '顿', '政', '府', '对', '威', '士', '忌', '酒', '暴', '乱', '的', '镇', '压', '得', '到', '▁', '了', '▁', '广', '泛', '的', '认', '同', '。', '[SEP]']
        lexicon_tree: 词典树
        max_word_num: 最多匹配的词的数量
    Return:
        sent_words: 句子中每个字归属的词组, 如：[[], ['华盛', '华盛顿'], ['华盛', '华盛顿'], ['华盛顿'], ['政府'], ['政府'], ['对威'], ['对威', '威士', '威士忌', '威士忌酒'], ['威士', '威士忌', '威士忌酒'], ['威士忌', '威士忌酒', '忌酒'], ['威士忌酒', '忌酒'], ['暴乱'], ['暴乱', '乱的'], ['乱的'], ['镇压'], ['镇压', '压得'], ['压得', '得到'], ['得到'], ['▁'], ['了'], ['▁'], ['广泛'], ['广泛'], ['的'], ['认同'], ['认同'], ['。'], []]
        sent_boundaries: 句子中每个字所属的边界类型3表示单字,0表示该字为词的开头, 如：[3, 0, 6, 2, 0, 2, 0, 5, 6, 7, 2, 0, 5, 2, 0, 5, 5, 2, 3, 3, 3, 0, 2, 3, 0, 2, 3, 3]
        以上每个元素对应于sent的每个元素。
    """
    sent_length = len(sent)
    sent_words = [[] for _ in range(sent_length)]
    sent_boundaries = [[] for _ in range(sent_length)]  # each char has a boundary

    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]  # speed using max depth
        words = lexicon_tree.enumerateMatch(sub_sent)

        if len(words) == 0 and len(sent_boundaries[idx]) == 0: #没有匹配到的单字
            sent_boundaries[idx].append(3) # S-
        else:
            if len(words) == 1 and len(words[0]) == 1: # 匹配到的单字 single character word
                if len(sent_words[idx]) == 0:
                    sent_words[idx].extend(words)
                    sent_boundaries[idx].append(3) # S-
            else:
                if max_word_num:
                    need_num = max_word_num - len(sent_words[idx])
                    words = words[:need_num]
                sent_words[idx].extend(words)
                for word in words:
                    if 0 not in sent_boundaries[idx]:
                        sent_boundaries[idx].append(0) # S-
                    start_pos = idx + 1
                    end_pos = idx + len(word) - 1
                    for tmp_j in range(start_pos, end_pos):
                        if 1 not in sent_boundaries[tmp_j]:
                            sent_boundaries[tmp_j].append(1) # M-
                        sent_words[tmp_j].append(word)
                    if 2 not in sent_boundaries[end_pos]:
                        sent_boundaries[end_pos].append(2) # E-
                    sent_words[end_pos].append(word)

    assert len(sent_words) == len(sent_boundaries)

    new_sent_boundaries = []
    idx = 0
    for boundary in sent_boundaries:
        if len(boundary) == 0:
            print("Error")
            new_sent_boundaries.append(0)
        elif len(boundary) == 1:
            new_sent_boundaries.append(boundary[0])
        elif len(boundary) == 2:
            total_num = sum(boundary)
            new_sent_boundaries.append(3 + total_num)
        elif len(boundary) == 3:
            new_sent_boundaries.append(7)
        else:
            print(boundary)
            print("Error")
            new_sent_boundaries.append(8)
    assert len(sent_words) == len(new_sent_boundaries)

    return sent_words, new_sent_boundaries

