## Non-parallel text style transfer with domain adaptation and an attention model (include adversarial network)
codes have been uploading and bebugged, welcome to learn my paper (doi: 10.1007/s10489-020-02077-5) though the model in it is not very advanced.)
the needed packages: tensorflow(1.12 - 1.15), nltk, sklearn ....
### The relevant papers are 
   1.  [Style transfer from non-parallel text by cross-alignment](https://arxiv.org/pdf/1705.09655.pdf) 
   code1:[go](https://github.com/shentianxiao/language-style-transfer) code2:[go](https://github.com/cookielee77/DAST/tree/master/network)

   2.  [Unpaired sentiment-to-sentiment translation: A cycled reinforcement learning approach](https://arxiv.org/pdf/1805.05181.pdf) 
   code:[go](https://github.com/lancopku/unpaired-sentiment-translation)
                        
   3.  [Toward controlled generation of text](https://arxiv.org/pdf/1703.00955.pdf) office code:[go](https://github.com/asyml/texar/tree/master/examples/text_style_transfer) 
   unoffice code:[go](https://github.com/cookielee77/DAST/tree/master/network)
                 
   4.  [Disentangled representation learning for non-parallel text style transfer](https://www.aclweb.org/anthology/P19-1041.pdf) 
   code:[go](https://github.com/vineetjohn/linguistic-style-transfer)

   5.  [Domain adaptive text style transfer](https://arxiv.org/pdf/1908.09395.pdf)
   code:[go](https://github.com/cookielee77/DAST/tree/master/network)
   
   
### supplement
sorry that the primitive py-file, that for extracting style words, has loss in the computer server of my school (i left the project files for a long time and the computer server was fixed once), then i write a new file to get the style words with the frequency ratio, however the WO score promote to 0.763 from 0.588 in yelp corpus. Thus if you do contrast experiments with other frameworks, you should keep the style words coincident for evaluation.
### Start
first: to download the yelp and imdb dataset from [the repository](https://github.com/cookielee77/DAST/tree/master/network) and put it to the data folder (you can use data_filter.py and data_filter_2.py to preprocess data), download the glove 100d for the validation path in config.py with website (http://nlp.stanford.edu/data/glove.6B.zip);
you can utilize the get_style_words.py to get the style words (you can also utilize other methods to obtain the style words)


second: to run data_filter.py to filter the imdb dataset;

third: to download glove 300d 100d, and add the path in the corresponding parameters of config.py and classifier/style.py, classifier/domain.py;

forth: to add relative parameters of mode paths, training;

fifth: to run classifier/style-classifier.py and domain-classifier.py to train  style and domain classifiers to evaluate the transferred sentences, and give the corresponding path in the config.py;

sixth: to run start-train.py for transferring sentences
       
### Acknowledge
The publication of the paper cannot leave with the helps and suggestions of the reviewers, they guide me how to improve my paper well; and my teacher Min He is a good teacher to support me to research new direction. And i am appreciate the journal -- applied intelligence which provide the chance for me to publish my paper. Lastly, i am very grateful for the researchers of style transfer, the public codes and the papers of them let me learn a lot.
