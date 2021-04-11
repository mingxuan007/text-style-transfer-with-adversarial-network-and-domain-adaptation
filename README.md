## Non-parallel text style transfer with domain adaptation and an attention model (include adversarial network)
codes have been uploading and bebugged, welcome to learn my paper though the model in it is not very advanced.)
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
   
### Start
first: to download the yelp and imdb dataset from [the repository](https://github.com/cookielee77/DAST/tree/master/network) and put it to the data folder;

second: to run data_filter.py to filter the imdb dataset;

third: to download glove 300d 100d, and add the path in the corresponding parameters of config.py and classifier/style.py, classifier/domain.py;

forth: to add relative parameters of mode paths, training;

fifth: to run classifier/style.py and domain train  style and domain classifiers to evaluate the transferred sentences;

sixth: to run start-train.py for transferring sentences
       
### Acknowledge
The publication of the paper cannot leave with the helps and suggestions of the reviewers, they guide me how to improve my paper well; and my teacher Min He is a good teacher to support me to research new direction. And i am appreciate the journal -- applied intelligence which provide the chance for me to publish my paper. Lastly, i am very grateful for the researchers of style transfer, the public codes and the papers of them let me learn a lot.
