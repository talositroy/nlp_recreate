import nltk

file = open("input.txt", "r")
text = file.read()
# 单词分割
words = nltk.word_tokenize(text)
print(words)
print('---------')
# 句子分割
sents = nltk.sent_tokenize(text)
print(sents)

