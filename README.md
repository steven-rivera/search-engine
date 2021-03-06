## Search Engine
I built a complete search engine that uses porter stemming, TF-IDF scoring, and weighting based on HTML tags to retrieve high relevance web pages from a corpus of over 55,000 web pages. Before the search engine can be run, the corpus must be indexed (indexer.py). I utilized an inverted index to index all the unique tokens that appear in the entire corpus storing information such as term-frequency, doc-frequency and a list of all the documents in which a particular term appears. In order to achieve a search result is less than 300ms I also create an index of the index during the indexing process. After indexing is complete the search engine can be run (searcher.py) allowing the user to enter a query in the terminal, after which the top 5 URL's will be displayed to the user.     


## How to Run

1.  **Setup**
      - External Packages
          - Install packages:
    	   	- `pip install beautifulsoup4 nltk`
    	
      - Edit Config File (config.json)
    	  - Set CORPUS_PATH to path of corpus folder
    	    -  `"CORPUS_PATH" : "/path/to/corpus"`
        
        - Set INDEX_STORAGE to path of folder that will store all indexes
          - `"INDEX_STORAGE" : "/path/to/index/storage"`
   
2. **Running Indexer**
    - `python indexer.py`


3. **Running Search Engine**
    - `python searcher.py`
