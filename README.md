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
