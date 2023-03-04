# g2pL
This is the official repository of our paper G2PL:LEXICON ENHANCED CHINESE POLYPHONE DISAMBIGUATION WITH A NEW
DATASET

## Install
```
pip install g2pL
```

## Requirements
```
pip install -r requirements.txt
```

## Download
[点击此处](https://github.com/whzikaros/g2pL/releases/tag/v0.0.1)下载模型及词向量等相关文件，将这些文件放入g2pL_files目录中。

## Usage
You can set the tone style, the default tone style is 2.
```
from G2pL import G2PL

>>> g2p = G2PL()
>>> g2p("请您把这台收音机的音量调低一点。", 0)
['qing', 'nin', 'ba', 'zhe', 'tai', 'shou', 'yin', 'ji', 'de', 'yin', 'liang', 'tiao', 'di', 'yi', 'dian', '。']
>>> g2p("请您把这台收音机的音量调低一点。", 1)
['qǐng', 'nín', 'bǎ', 'zhè', 'tái', 'shōu', 'yīn', 'jī', 'de', 'yīn', 'liàng', 'tiáo', 'dī', 'yī', 'diǎn', '。']
>>> g2p("请您把这台收音机的音量调低一点。", 2)
['qing3', 'nin2', 'ba3', 'zhe4', 'tai2', 'shou1', 'yin1', 'ji1', 'de5', 'yin1', 'liang4', 'tiao2', 'di1', 'yi1', 'dian3', '。']

```
