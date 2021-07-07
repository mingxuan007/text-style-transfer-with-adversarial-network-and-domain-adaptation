corpus_path = ""                            #the train path of the corpus
sentiment_path = ""                         #the style_words_file to save 
f = open(corpus_path,"r").readlines()
f_sentiment = open(sentiment_path,"w")
import json
set_0 = []
set_1 = []
for line in f:
    line_dict = json.loads(line)
    if line_dict["score"] == 0:
       set_0 += line_dict["review"].split()
    if line_dict["score"] == 1:
       set_1 += line_dict["review"].split()
from collections import Counter
import math
word_set_0 = Counter(set_0)
word_set_1 = Counter(set_1)

# print(word_set_0)
# print(word_set_1)
word_set = set(word_set_0 + word_set_1)
word_set_0_fre = dict()
word_set_1_fre = dict()
for word in word_set:
    # print(word)
    # print((word_set_0[word]+1) // (word_set_1[word]+1)+1)
    # print(math.log(2))
    word_set_0_fre[word] = math.log((word_set_0[word]+1) // (word_set_1[word]+1)+1)
    word_set_1_fre[word] = math.log((word_set_1[word]+1) // (word_set_0[word]+1)+1)

word_set_0_fre = sorted(word_set_0_fre.items(),key=lambda i:i[1],reverse=True)
word_set_1_fre = sorted(word_set_1_fre.items(),key=lambda i:i[1],reverse=True)
word_set_0_fre = [word_set[0] for word_set in word_set_0_fre[:3500]]
word_set_1_fre = [word_set[0] for word_set in word_set_1_fre[:3500]]

p=0

# we only extract 1000 style words, you can decide to the number with the reasonable situation
for word in word_set_1_fre:
    f_sentiment.write(word + "\n")
    p += 1
    if p == 500:
       p = 0
       break
for word in word_set_0_fre:
    f_sentiment.write(word + "\n")
    p += 1
    if p == 500:
        p = 0
        break
