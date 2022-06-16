from collections import defaultdict
from pathlib import Path
from bs4 import BeautifulSoup
from math import log10
import tokenizer
import json
import sys

# INVERTED INDEX JSON STRUCTURE
# {
#   "TOKEN": {
#       "docFrequency": int,
#       "postingList": [
#            {
#               "docID": int,
#               "tokenFrequency": int,
#               "tokenImportance" : int,
#               "tf_idf": float
#             },
#        ]
#   }         
# }


# STATIC GLOBAL VARIABLES
DEBUG = True
CONFIG_FILE = "config.json"
MAX_INDEX_SIZE = 5_000_000 # 5 MB
TOKEN_IMPORTANCE = {
    "title": 10,
    "h1": 5,
    "h2": 4,
    "h3": 3,
    "b": 2,
    "strong": 2
}


# GLOBAL VARIABLES
invertedIndex = {}     # Will be used to store the main index
currPartialIndex = 1   # Keeps track of how many partial indexes have been created (ex. if currPartialIndex=5 then partial indexes 1-4 have been created)
numberOfDocs = 0       # Keeps track of how many docs have been indexed
currDocID = 0          # Assigns docID's to documents in the order that they are indexed
docIDtoURL = []        # List containing URL's each document where docID is the index containing the URL




def createPostingsForDocument(html: str, docID: int) -> dict[str, dict[str, int]]:
    """
    Returns a dictionary where the key is a token and the value
    is a posting for the current document docID. The posting is of
    the format,

    {
     "docID": int,
     "tokenFrequency": int,
     "tokenImportance" : int,
     "tf_idf": float
    }

    """
    
    soup = BeautifulSoup(html, "lxml")
    tokens = tokenizer.tokenize(soup.get_text())
    frequencies = tokenizer.computeWordFrequencies(tokens)

    # Create a dict where the key is a token and 
    # the value is the token's importance rating
    importantTokens = {}
    for tagObject in soup.find_all(["b", "strong", "h1", "h2", "h3", "title"]):
        for token in tokenizer.tokenize(str(tagObject.string)):
            importantTokens[token] = TOKEN_IMPORTANCE.get(tagObject.name, 1)
                

    postings = defaultdict(dict)
    
    for token, frequency in frequencies.items():
        postings[token]["docID"]           = docID
        postings[token]["tokenFrequency"]  = frequency
        postings[token]["tokenImportance"] = importantTokens.get(token, 1)  # Set to 1 if token not in importantTokens
        postings[token]["tf_idf"]          = 0.0  # td_idf we be calculated during later step, initialize to 0.0

    return postings

 




def writeURLsToDisk(docIDtoURL: list[str]) -> None:
    """
    Saves the URL's to disk as a text file. Line N
    corresponds the the URL with docID N-1. 
    (Ex. Line=70 => docID=69)
    """
    
    with open(CONFIG_FILE) as f:
        # Gets path of folder to save index
        indexFolderPath = json.load(f)["INDEX_STORAGE"]

    # Slash operator combines paths and creates new Path object to docIDtoURL file  
    docIDtoURL_filePath = Path(indexFolderPath) / "docIDtoURL.txt"

    if DEBUG: print(f"\n{f'WRITING docIDtoURL.txt':=^{40}}")

    with docIDtoURL_filePath.open(mode="w", encoding="utf-8") as f:
        for url in docIDtoURL:
            f.write(f"{url}\n")

    if DEBUG: print(f"{f'CREATED docIDtoURL.txt':=^{40}}\n")






def writePartialIndexToDisk(invertedIndex: dict[str, dict]) -> None:
    """
    Saves the current invertedIndex to a partialIndex file
    which will later be used to merge into one full index file.
    Each line contains a INVERTED INDEX JSON STRUCTURE containing
    only one token. The tokens are sorted in ascending order.
    """
    global currPartialIndex

    with open(CONFIG_FILE) as f:
        # Gets path of folder to save index file
        indexFolderPath = json.load(f)["INDEX_STORAGE"]

    # Slash operator combines paths and creates new Path object to partial index 
    partialIndexFilePath = Path(indexFolderPath) / f"partialIndex_{currPartialIndex}.txt"

    if DEBUG: print(f"\n{f'WRITING partialIndex_{currPartialIndex}.txt':=^{40}}")
    
    with partialIndexFilePath.open("w", encoding="utf-8") as f:
        # Save tokens to file in sorted order to speed merging process
        for token in sorted(invertedIndex):
            # Dump inverted index JSON structure onto its 
            # own line containing only a single term
            fileLine = {f"{token}": invertedIndex[token]}
            f.write(f"{json.dumps(fileLine)}\n")
    
    if DEBUG: print(f"{f'CREATED partialIndex_{currPartialIndex}.txt':=^{40}}\n")

    # Keeps track of how many partial indexs have been created
    currPartialIndex += 1





def iterCorpus() -> Path:
    """
    Is a generator function that yields a path object of the 
    current JSON file to be indexed.
    """

    with open(CONFIG_FILE) as f:
        # Gets path of DEV folder
        corpus_path = json.load(f)["CORPUS_PATH"]
        
    
    corpus_path = Path(corpus_path)
    for folder in corpus_path.iterdir():
        for document in folder.iterdir():
            yield document




def merge(file1: Path, file2: Path, mergedFile: Path) -> None:
    """
    Merges two partial index files into the given mergedFile path.
    The mergere preserves the alphabetical ordering of tokens.
    """

    f1, f2 = file1.open(mode="r"), file2.open(mode="r")

    with mergedFile.open(mode="w", encoding="utf-8") as f3:
        
        f1_line, f2_line = f1.readline().strip(), f2.readline().strip()
        while (f1_line != "") and (f2_line != ""):
            
            # Convert current line of each file into dictionary object
            f1_line_json, f2_line_json = json.loads(f1_line), json.loads(f2_line)
            
            # Get token from dictionary object
            f1_token, f2_token = list(f1_line_json.keys())[0], list(f2_line_json.keys())[0]

        
            if (f1_token == f2_token):
                # If tokens are the same then merge objects into f1_line dictionary
                f1_line_json[f1_token]["docFrequency"] += f2_line_json[f1_token]["docFrequency"]
                f1_line_json[f1_token]["postingList"].extend(f2_line_json[f1_token]["postingList"]) 

                # Write merged token to file
                f3.write(f"{json.dumps(f1_line_json)}\n")

                # Read next line from both files
                f1_line, f2_line = f1.readline().strip(), f2.readline().strip()
            
            # Done to maintian alphabetical order 
            elif (f1_token < f2_token):
                f3.write(f"{json.dumps(f1_line_json)}\n")
                f1_line = f1.readline().strip()
            
            else:
                f3.write(f"{json.dumps(f2_line_json)}\n")
                f2_line = f2.readline().strip()

        
        # Finish writing f1 to f3 if there are no more lines from f2
        while (f1_line != ""):
            f3.write(f"{f1_line}\n")
            f1_line = f1.readline().strip()

        # Finish writing f2 to f3 if there are no more lines from f1
        while (f2_line != ""):
            f3.write(f"{f2_line}\n")
            f2_line = f2.readline().strip()


    # Close files once done reading from them
    f1.close()
    f2.close()




    

def mergePartialIndexes() -> None:
    """
    Merges all of the partial index files into a single index
    file called 'index.txt'. Total file merges is O(N)
    where N is the number of partial indexes.

    Examples:

        Case 1:

            partialIndex_1.txt    partialIndex_2.txt    partialIndex_3.txt    partialIndex_4.txt
                            \            /                           \            /
                            partialIndex_5.txt                     partialIndex_6.txt
                                                    \       /
                                                    index.txt

        Case 2:

            partialIndex_1.txt    partialIndex_2.txt    partialIndex_3.txt
                            \            /                   /            
                            partialIndex_4.txt              /                        
                                                   \       /
                                                   index.txt
    """

    with open(CONFIG_FILE) as f:
        # Gets path of folder containg partial index files
        indexFolderPath = json.load(f)["INDEX_STORAGE"]
    


    def _mergePartialIndexes() -> None:
        """
        Private function which recursively merges all of the 
        partial index files into a single index file.
        """
        global currPartialIndex


        partialIndexPaths = [path for path in Path(indexFolderPath).iterdir() if path.name.startswith("partialIndex_")]

        if len(partialIndexPaths) == 1:
            # All files have been merged
            # Rename file to index.txt
            partialIndexPaths[0].replace(Path(indexFolderPath) / "index.txt")
            return

        partialIndexIterator = iter(partialIndexPaths)
        while True:
            try:
                fileToMerge_1 = next(partialIndexIterator)
                fileToMerge_2 = next(partialIndexIterator)
                
                # Slash operator combines paths and creates new Path object to partial index 
                mergedFilePath = Path(indexFolderPath) / f"partialIndex_{currPartialIndex}.txt"
                merge(fileToMerge_1, fileToMerge_2, mergedFilePath)
                currPartialIndex += 1

                # Delete partial indexes since they are now merged
                fileToMerge_1.unlink()
                fileToMerge_2.unlink()
                
            except StopIteration:
                break

        _mergePartialIndexes()
    
    _mergePartialIndexes()

        



def createPartialIndexes() -> None:
    global invertedIndex
    global docIDtoURL
    global currDocID
    global numberOfDocs

    
    for document in iterCorpus():
        with document.open() as f:
            jsonData = json.load(f)
            
            # Save URL to associate docID with URL
            url = jsonData.get("url", "")
            docIDtoURL.append(url)
            
            if DEBUG: print(f"INDEXDING docID={currDocID}, INDEXSIZE={sys.getsizeof(invertedIndex):,} Bytes, URL={url}")
    
            # Retrieve and parse htlm of current document
            html = jsonData.get("content", "")
            postings = createPostingsForDocument(html, currDocID)

            for token, posting in postings.items():
                if token in invertedIndex:
                    # If token has been seen then increment docFreq
                    # and append new posting to postingList
                    invertedIndex[token]["docFrequency"] += 1
                    invertedIndex[token]["postingList"].append(posting)
                else:
                    # Initialze the postingList for new token and 
                    # set docFreq to 1
                    invertedIndex[token] = {
                        "docFrequency" : 1,
                        "postingList" : [posting]
                    }
            
            currDocID += 1
            numberOfDocs += 1


            if sys.getsizeof(invertedIndex) > MAX_INDEX_SIZE:
                # Save partial index to disk to save memory 
                writePartialIndexToDisk(invertedIndex)
                # Reset invertedIndex to empty dictionary
                invertedIndex.clear() 

    

    # Save last part of index to disk
    writePartialIndexToDisk(invertedIndex)
    # Save docIDtoURL mapping to disk
    writeURLsToDisk(docIDtoURL)




def update_tf_idf_score() -> None:
    """
    Iterates through the merged index and calculates the tf_idf score
    for every posting in every postingList. Writes updated index back
    to disk.
    """

    global numberOfDocs
    
    with open(CONFIG_FILE) as f:
        # Gets path of folder to save index file
        indexFolderPath = json.load(f)["INDEX_STORAGE"]

    # Slash operator combines paths and creates new Path object 
    oldIndex = Path(indexFolderPath) / "index.txt"
    updatedIndex = Path(indexFolderPath) / "temp.txt"

    # Open files
    old = oldIndex.open(mode="r", encoding="utf-8")
    updated = updatedIndex.open(mode="w", encoding="utf-8")

    for line in old:
        # Convert text line to json object
        line_dict = json.loads(line.strip())
        
        token = list(line_dict.keys())[0]
        docFrequency = line_dict[token]["docFrequency"]

        for posting in line_dict[token]["postingList"]:
            tf = posting["tokenFrequency"]
            posting["tf_idf"] = ( 1 + log10(tf) ) * log10(numberOfDocs/docFrequency)

        updated.write(f"{json.dumps(line_dict)}\n")
        
   
    # Close files
    old.close()
    updated.close()

    # Delete old index file
    oldIndex.unlink()

    # Rename updated index from 'temp.txt' to 'index.txt'
    updatedIndex.replace(Path(indexFolderPath) / "index.txt")





def createIndexofIndex():
    """
    Indexes the character position of every token in
    the main index and saves index of index to file 'indexOfIndex.txt'
    """

    with open(CONFIG_FILE) as f:
        # Gets path of folder to save index file
        indexFolderPath = json.load(f)["INDEX_STORAGE"]

    # Slash operator combines paths and creates new Path object to indexFile
    indexFilePath = Path(indexFolderPath) / "index.txt"
    
    indexOfIndex = dict()
    with indexFilePath.open(mode="r", encoding="utf-8") as indexFile:
        while True:
            # Get character position of current line
            seekPosition = indexFile.tell()
            line = indexFile.readline().strip()

            if line == "":
                break

            # Convert text line to json object
            line_dict = json.loads(line.strip())
            token = list(line_dict.keys())[0]

            indexOfIndex[token] = seekPosition


    # Save index of index to disk with file name 'indexOfIndex.txt'
    indexOfIndexFilePath = Path(indexFolderPath) / "indexOfIndex.txt"

    with indexOfIndexFilePath.open(mode="w", encoding="utf-8") as indexOfIndexFile:
        for token, position in indexOfIndex.items():
            indexOfIndexFile.write(f"{token} {position}\n")





def main() -> None:

    if DEBUG: print(f"{'CREATING PARTIAL INDEXES':=^{40}}\n")
    createPartialIndexes()
    if DEBUG: print(f"\n{'FINISHED PARTIAL INDEXES':=^{40}}")

    if DEBUG: print(f"PARTIAL INDEXES CREATED => {currPartialIndex-1}")
    if DEBUG: print(f"DOCUMENTS INDEXED => {numberOfDocs}\n")


    if DEBUG: print(f"\n{'MERGING PARTIAL INDEXES':=^{40}}")
    mergePartialIndexes()
    if DEBUG: print(f"{'MERGE COMPLETE, MAIN INDEX CREATED':=^{40}}\n")

    if DEBUG: print(f"\n{'CALCULATING TD-IDF SCORE':=^{40}}")
    update_tf_idf_score()
    if DEBUG: print(f"{'FINISHED TD-IDF SCORING':=^{40}}\n")

    if DEBUG: print(f"\n{'CREATING INDEX OF INDEX':=^{40}}")
    createIndexofIndex()
    if DEBUG: print(f"{'FINISHED INDEX OF INDEX':=^{40}}")


if __name__ == "__main__":
    main()