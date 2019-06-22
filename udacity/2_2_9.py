from nltk.tokenize import word_tokenize

file = open("input.txt", "r")
text = file.read()
words = word_tokenize(text)
print(words)
