import random
import numpy as np
import torch
import os
import shutil
import logging
import pickle
# from sklearn.metrics import f1_score
# import thulac

def set_seed(seed,gpu_num):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    if gpu_num > 0:
        torch.cuda.manual_seed_all(seed)

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

#查看模型架构
def show_model(model):
    for name, module in model.named_children():
        if name == "bert":
            for n, _ in module.named_children():
                print(f"{name}:{n}")
        else:
            print("{:15} {}".format(name, module))

def accuracy(preds, labels):
    return (preds == labels).mean()

# def acc_and_f1(preds, labels):
#     acc = accuracy(preds, labels)
#     f1 = f1_score(y_true=labels, y_pred=preds)
#     return {
#         "acc": acc,
#         "f1": f1,
#         "acc_and_f1": (acc + f1) / 2,
#     }
    
def load_pkl(file):
    """
    读取pkl文件
    """
    with open(file, "rb") as f:
        class2idx = pickle.load(f)
    
    print("class2idx: ",class2idx)    
    return class2idx

def readlines(file):
    """
    按行读取文本文件
    """
    ensure_nod(file)
    with open(file, 'r', encoding="utf-8") as f:
        data_list=f.readlines()
    return data_list

def writelines(file):
    """
    按行读取文本文件
    """
    ensure_nod(file)
    with open(file, 'r', encoding="utf-8") as f:
        f.writelines()
    print("writelines()函数保存成功！")
    
def ensure_dir(path):
    """ 确保目录存在 """
    if not os.path.exists(path):
        os.makedirs(path)
        
def ensure_nod(path):
    """ 确保文件存在 """
    if not os.path.exists(path):
        os.mknod(path)

def remove_dirs(path):
    """如果文件夹存在则移除该文件夹"""
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
def remove_file(path):
    """如果文件存在则移除该文件"""
    if os.path.exists(path):
        os.remove(path)
def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def comput_txt_num(file):
    with open(file,'r') as f:
        data=f.readlines()
    print("文本数据量：{}".format(len(data)))
    
    
# def thulac_tokenize(text,thulac_tokenzier):
#     """
#     分词
#     参数：thulac_tokenzier为分词器，默认等于thulac.thulac(user_dict=None)
#     """
    
#     words = []
#     pos = []
#     pairs = thulac_tokenzier.cut(text)
#     for pair in pairs:
#         words.append(pair[0])
#         pos.append(pair[1])
#     return words, pos

if __name__=="__main__":
    pkl_file="/disc1/hongzhi.wan/my_polyphone/data/class2idx.pkl"
    file1="/disc1/hongzhi.wan/my_polyphone/make_data/database/baike_qa2019/content2.txt"
    file2="/disc1/hongzhi.wan/my_polyphone/make_data/database/new2016zh/content2.txt"
    file3="/disc1/hongzhi.wan/my_polyphone/make_data/database/translation2019zh/content2.txt"
    file4="/disc1/hongzhi.wan/my_polyphone/make_data/database/webtext2019zh/content2.txt"
    file5="/disc1/hongzhi.wan/my_polyphone/make_data/database/wiki_zh_2019/content2.txt"
    output_file="/disc1/hongzhi.wan/my_polyphone/make_data/database/all_text.txt"
    #load_pkl(pkl_file)
    comput_txt_num(file1)
    comput_txt_num(file2)
    comput_txt_num(file3)
    comput_txt_num(file4)
    comput_txt_num(file5)
    comput_txt_num(output_file)