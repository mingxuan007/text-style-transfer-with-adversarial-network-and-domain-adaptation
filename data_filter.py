from config import load_arguments
import tensorflow as tf
import json
import copy
args=load_arguments()
target_path = args.target_train_path+"/train.txt"
f=open(target_path,'r')
text_tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters='!"#$%&()*+-./:=?@[\\]^_`{|}~\t\n')
text_tokenizer.fit_on_texts(f)
word_index=text_tokenizer.word_index
source_path = args.source_train_path+"/train.txt"
text=[]

for i in open(source_path,'r').readlines():
    text.append(i)
text=copy.deepcopy(text)
fy=open(source_path,'w')
print(len(text))
for te in text:

    p=0
    re=json.loads(te.strip())["review"]
    re=re.split()
    for i in re:
        if i not in word_index.keys():
           print(i)
           p=1
           break

    if p == 0:
       print(te)
       fy.write(te)
