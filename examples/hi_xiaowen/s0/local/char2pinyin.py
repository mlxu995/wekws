from tqdm import tqdm
from pypinyin import pinyin, Style
from pypinyin.contrib.tone_convert import to_initials, to_finals_tone3


txt_file = '/home/mlxu/github/wekws/examples/hi_xiaowen/s0/data_ctc/test/text'
pinyin_file = '/home/mlxu/github/wekws/examples/hi_xiaowen/s0/data_ctc/test/text_pinyin'

with open(txt_file) as f, open(pinyin_file, 'w') as f_w:
    for line in tqdm(list(f.readlines())):
        key, *words = line.strip().split()
        pinyin_list = []
        for word in words:
            pinyin_list.extend(pinyin(word, style=Style.TONE3, heteronym=True))
        pinyin_list = [to_initials(c[0]) + ' ' + to_finals_tone3(c[0]) for c in pinyin_list]
        f_w.write(key + ' ' + ' '.join(pinyin_list) + '\n')