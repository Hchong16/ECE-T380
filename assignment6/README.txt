#Harry Chong
I wanted to load the most recent wikipedia dataset(20200301.en), but I kept receiving an error message
regarding Apache Beam and how the dataset is too large. In order for me to get around this issue, I had 
to downgrade my tensorflow_datasets package from version 4.1.0 to version 2.0.0. This allowed me to load 
the dataset into jupyter notebook.

Additionally, I was not sure how to estimate the top 20 most likely titles for an article text. Instead,
you can run the topTwentySentences.py to run the trained model and see the predictions it made for a number 
of wikipedia articles. There is a parameter named num_articles (on line 298) where you can adjust the number 
of wikipedia articles to pass into the model. 

The topTwentySentences.py file might take a few minutes before it starts outputting results since it has to take 
a large portion of the dataset and tokenize it.

Thank you!
