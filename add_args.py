import argparse
def add_arg(parser):
    parser.add_argument(
        "--model_name_or_path",
        default="g2pL/bert_config",
        type=str,
        help="使用已预训练的模型名及路径名。即pytorch_model.bin所在的路径。")
    parser.add_argument(
        "--config_name",
        # default="bert_model_files/chinese_bert_base/config_lebert.json",
        default="g2pL/bert_config/config_g2pL.json",
        type=str,
        help="bert预训练模型的设置文件所在路径。即config.json所在的路径。")
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="需要训练的文件")
    parser.add_argument(
        "--train_label_file",
        default=None,
        type=str,
        help="需训练的标签文件")
    parser.add_argument(
        "--val_file",
        default=None,
        type=str,
        help="需要验证的文件")
    parser.add_argument(
        "--val_label_file",
        default=None,
        type=str,
        help="需验证的标签文件")
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        help="需要测试的文件")
    parser.add_argument(
        "--test_label_file",
        default=None,
        type=str,
        help="需测试的标签文件")
    parser.add_argument(
        "--class2idx_file",
        default="g2pL/class2idx.pkl",
        type=str,
        help="标签id序列字典文件")
    parser.add_argument(
        "--test_model",
        default="g2pL_files/best_model.pt",
        type=str,
        help="用于test的pt模型路径")    
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="输出文件所在的路径。")   
    parser.add_argument(
        '--log_terminal_file', 
        default="output/train.log", 
        help="记录训练输出到终端的内容。")
            
    #需要开启的参数
    parser.add_argument(
        "--do_train", 
        action="store_true", 
        help="Whether to run training.")
    parser.add_argument(
        "--do_eval", 
        action="store_true", 
        help="不边训练边eval的时候可以开这个,默认不开。")
    parser.add_argument(
        "--do_test", 
        action="store_true", 
        help="Whether to run test on the test set.")    
    parser.add_argument(
        "--evaluate_during_training", 
        action="store_true", 
        help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--no_cuda", 
        action="store_true", 
        help="不显示则表示使用cuda,Whether not to use CUDA when available"
    )
    
    #超参数设置
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=320)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    help="分词后的序列长度，短了则pad补齐，长了则截断。")
    
    #其他设置
    parser.add_argument(
        "--n_gpu", 
        type=int, 
        default=1, 
        help="gpu数量.")
    parser.add_argument(
        "--logging_ratio", 
        type=float, 
        default=0.1, 
        help="Log every X updates ratio.")
    parser.add_argument(
        "--save_ratio", 
        type=float, default=0.1, 
        help="Save checkpoint every X updates ratio.")    

    #预处理参数
    parser.add_argument(
        "--word_vocab_file",
        default="data/vocab/tencent_vocab.txt",
        type=str,
        help="制作词典树的来源，也是腾讯的预训练word_embedding的词表。")
    parser.add_argument(
        "--max_scan_num", 
        type=int, 
        default=1000000 , 
        help=".")
    parser.add_argument(
        "--word_embedding",
        default="G2pL/embedding/word_embedding.txt",
        type=str,
        help="预训练的word_embedding，也是腾讯的预训练word_embedding的词表。")   
    parser.add_argument(
        "--word_embed_dim", 
        type=int, 
        default=200, 
        help="词向量的维度.")    
    parser.add_argument(
        "--saved_embedding_dir",
        default="data/embedding",
        type=str,
        help="保存的embedding的路径。")   
    parser.add_argument("--max_word_num", default=5, type=int, help="最多匹配的词的数量。")     

            
    args = parser.parse_args()
    return args