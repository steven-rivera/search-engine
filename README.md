# Search Engine

I built a lightweight yet powerful search engine in Python capable of processing a web corpus of over **100,000+** documents and responding to queries in under **300ms**. It utilizes **porter stemming**, **TF-IDF** scoring, and HTML tag-based **weighting** to retrieve the most relevant web pages from the entire corpus for a given query.

<div align="center">
  <img src="images/search-engine.png" width="75%">
</div>

## Project Overview

This project is composed of two main components:

1. `indexer.py`:  Indexes a corpus of HTML documents.
2. `searcher.py`: Queries the indexed data and returns the URLs of the most relevant results.


Before the search engine can be run, the corpus must be **indexed** in order to efficiently respond to queries. The search engine utilizes an **inverted index** which maps tokens to a list of documents containing the token. You can read more about the indexing process [here](#Indexing-Process). Once the corpus is indexed, you can query the corpus in which the search engine will return the top 5 most relevant URL's pertaining to the query.

This repository provides a small corpus of documents located in [`CORPUS/`](CORPUS/) as an example of how the indexer expects the folder to be structured. To generate a larger more comprehensive corpus of documents you can use my custom [web crawler](https://github.com/steven-rivera/web-crawler). More information on how the corpus folder is structed can be found [here](https://github.com/steven-rivera/web-crawler?tab=readme-ov-file#directory-structure).

## Getting Started

### 1. Clone the Repository 

```bash
git clone https://github.com/steven-rivera/search-engine
cd search-engine
```

### 2. Create and Activate a Virtual Environment (Optional)

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat   # For Windows
source venv/bin/activate    # For macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Paths

Edit the `config.json` file to match your system:

```json
{
  "CORPUS_PATH": "/path/to/corpus",
  "INDEX_STORAGE": "/path/to/store/indexes"
}
```

- `CORPUS_PATH`: Folder containing corpus of HTML documents to be indexed.    
- `INDEX_STORAGE`: Folder to store index files.

> [!IMPORTANT]
> A small sample corpus is provided [here](CORPUS/) showing how the indexer expects the corpus to be structured. To generate your own corpus you can use my [web crawer](https://github.com/steven-rivera/web-crawler) program. 

### 5. Run the Indexer

The corpus **must** be indexed before the search engine can execute a query. After following the previous steps listed run:

```bash
python indexer.py
```

> [!NOTE]
> Indexing time will vary depending on the size of the corpus

### 6. Run the Search Engine

**Console Mode (default):**

```bash
python searcher.py
```

**Web Interface:**

```bash
python searcher.py --webapp
```

## Indexing Process

Before any searches can be performed, the corpus must be indexed. This allows the search engine to efficiently respond to user queries without having to parse the entire corpus for each query. The indexing process involves several key steps:

### 1. Document ID Assignment

Each document in the corpus is incrementally assigned an integer ID which maps to the corresponding URL of the document, starting with ID `0`. This reduces the size of the index on disk as the index only needs to store an integer to refer to a document rather than its entire URL. In order to preserve the ID to URL mapping, each documents' URL is saved to a text file where line `N+1` contains to the URL for the document with ID `N`. 

### 2. Inverted Index Construction

For each document processed, a **tokenizer** extracts and normalizes tokens contained within the document. These tokens are then used to build an **inverted index** where every unique token maps to a list of **postings**. Each posting includes:

- `Document ID`: The ID of the document where the token appeared.
- `Token Frequency (TF)`: How many times the token appeared in the document.
- `Token Importance`: An integer weight based on the HTML tag the token appeared in. For example:
    - `<title>`: highest weight
    - `<h1>`, ..., `<h6>`, : descending weights
    - `<p>`: lowest weight

The inverted index is stored on disk as a `JSONL` file,  with each line containing a `JSON` object with the following structure:

```json
{
  "TOKEN": [
    {
      "docID": 0,
      "tokenFrequency": 1,
      "tokenImportance": 10,
    }, 
  ]
}
```

> [!NOTE]
> When indexing a large corpus the size of the index may exceed the amount of memory available on the machine, causing the indexer to fail. To prevent this, the index is periodically saved to disk once it reaches a certain size. Once all the documents in the corpus are parsed, the indexer must merge all of these partial index files into a single unified index file before continuing to the next step.

### 3. TF-IDF Calculation

Once all documents have been processed, the **Term Frequency-Inverse Document Frequency (TF-IDF)** score for each token-document pair is calculated and index is updated. This score is what the search engine uses to rank documents based on a given query. You can read more about how the score is used and calculated [here](#document-ranking).

Once the TF-IDF score is calculated, it is no longer necessary to store the `tokenFrequency` and `tokenImportance` keys in the index. The updated index is stored on disk as a `JSONL` file as before, but with each line containing a `JSON` object with the following structure:

```json
{
  "TOKEN": [
    {
      "docID": 0,
      "tf_idf": 2.4727564493172123
    }, 
  ]
}
```

### 4. Meta-Index Creation

To avoid loading the entire inverted index into memory — which could be larger than the available memory on the machine — a **meta-index** is also created. It maps each token in the index to its **byte offset** within the index file on disk. This allows:

- Fast access to specific token postings using `seek()`
- Low memory usage during search time
- High scalability to large corpus