## Non-parallel text style transfer with domain adaptation and an attention model
codes have been uploading, and then i will modify relevant file routes for running these codes. (sorry, i recently have somethings to do, leading the upload be slow)
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
first: to download the yelp dataset from [the repository](https://github.com/cookielee77/DAST/tree/master/network) and put it to the data folder;
sencond: to run classifier/style-classifier.py to train a style classifier to evaluate the transferred sentences;
third: to run start-train.py for transferring sentences
       
