from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk

file = open("input.txt", "r")
text = file.read().lower()
# 单词分割
words = word_tokenize(text)
print('---------单词分割---------')
print(words)
# 句子分割
sents = sent_tokenize(text)
print('---------句子分割---------')
print(sents)
# 停用词
stop_word = stopwords.words("english")
print('---------停用词---------')
print(stop_word)
# 去除停用词
words_s_ed = [w for w in words if w not in stop_word]
print('---------去除停用词词---------')
print(words_s_ed)
# 词性标注
print('---------词性标注---------')
print(pos_tag(words))
# 命名实体识别
print('---------命名实体识别---------')
words2 = "Antonio joined Udacity Inc. in California."
ne_chunk(pos_tag(word_tokenize(words2)))
