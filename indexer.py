from typing import Generator
from collections import defaultdict
from pathlib import Path
from math import log10
import json, sys

import tokenizer
from color import red, green, yellow, grey

from bs4 import BeautifulSoup


# INVERTED INDEX JSON STRUCTURE

# Stage 1:
#   {
#     "TOKEN": [
#       {
#         "docID": int,
#         "tokenFrequency": int,
#         "tokenImportance": int,
#       }, 
#     ]
#   }

# Stage 2:
#   {
#     "TOKEN": [
#       {
#         "docID": int,
#         "tf_idf": float
#       }, 
#     ]
#   }


# GLOBAL VARIABLES
DEBUG                           = True
DEBUG_MESSAGE_WIDTH             = 50
CONFIG_FILE_NAME                = "config.json"
PARTIAL_INDEX_FILE_NAME_PREFIX  = "partialIndex"
INDEX_FILE_NAME                 = "index.jsonl"
INDEX_OF_INDEX_FILE_NAME        = "indexOfIndex.json"
DOCID_TO_URL_FILE_NAME          = "docIDtoURL.txt"
MAX_INDEX_SIZE                  = 5_000_000 # 5 MB
DEFAULT_IMPORTANCE              = 1
TAG_IMPORTANCE                  = {"title": 10, "h1": 5, "h2": 4, "h3": 3, "strong": 2}


class Indexer:
    def __init__(self, corpusFolderPath: Path, indexFolderPath: Path):
        self.corpusFolderPath = corpusFolderPath
        self.indexFolderPath = indexFolderPath

        self.index: dict[str, list] = {}  # Will be used to store the main index  
        self.docIDtoURL: list[str] = []   # List of URL's where docID is the index containing the URL for a given document
        self.numberOfDocsIndexed: int = 0 # Keeps track of how many docs have been indexed
        self.currPartialIndex: int = 1    # Keeps track of how many partial indexes have been created (ex. if currPartialIndex == 5 then partial indexes 1-4 have been created)
        

    def run(self):
        self.createPartialIndexes()
        self.mergePartialIndexesToDisk()
        self.calculateTF_IDF()
        self.createIndexofIndex()
        self.writeDocIdToURLToDisk()
            

    def iterCorpus(self) -> Generator[Path, None, None]:
        """
        Is a generator function that yields a path object of the current document to be indexed.
        The document is stored as a JSON file of the form:

        {
            "url": string,
            "content": string
        }
        """

        for folder in self.corpusFolderPath.iterdir():
            for document in folder.iterdir():
                if document.is_file():
                    yield document


    def createPartialIndexes(self) -> None:
        if DEBUG: print(f"{' CREATING PARTIAL INDEXES ':=^{DEBUG_MESSAGE_WIDTH}}")

        currDocID = 0
        for document in self.iterCorpus():
            with document.open() as f:
                try:
                    jsonData: dict[str, str] = json.load(f)
                except:
                    print(yellow(f"Warning: Invalid JSON file: {f.name}"))
                    continue

                url = jsonData.get("url", "")
                html = jsonData.get("content", "")

                if url == "" or html == "":
                    continue


                self.docIDtoURL.append(url)
                postings: dict[str, dict[str, int]] = self.parseDocument(html, currDocID)
                
                for token, posting in postings.items():
                    if token in self.index:
                        # Add new posting to postingList for given token
                        self.index[token].append(posting)
                    else:
                        # Initialze the postingList for new token
                        self.index[token] = [posting]

                self.numberOfDocsIndexed += 1

                if DEBUG and self.numberOfDocsIndexed % 10 == 0: 
                    print(grey(f"{self.numberOfDocsIndexed} docs indexed, Current Index Size: {sys.getsizeof(self.index):,} Bytes"))

            # Save partial index to disk if invertedIndex excedes memory capacity 
            if sys.getsizeof(self.index) > MAX_INDEX_SIZE:
                if DEBUG: print(yellow(f"Current Index Size greater than {MAX_INDEX_SIZE:,} Bytes\nSaving index to disk..."))
                self.writePartialIndexToDisk()
                self.index.clear() 

            currDocID += 1

        self.writePartialIndexToDisk()
        if DEBUG: print(f"Total Docs Indexed: {self.numberOfDocsIndexed}")

    def parseDocument(self, html: str, docID: int) -> dict[str, dict[str, int]]:
        """
        Returns a dictionary where the key is a token and the value
        is a posting for the current document docID. The posting is of
        the format:

        {
         "docID": int,
         "tokenFrequency": int,
         "tokenImportance": int,
        }
        """

        soup = BeautifulSoup(html, "lxml")
        tokens = tokenizer.tokenize(soup.get_text())
        frequencies = tokenizer.computeWordFrequencies(tokens)

        postings = defaultdict(dict)
        for token, frequency in frequencies.items():
            postings[token]["docID"]           = docID
            postings[token]["tokenFrequency"]  = frequency

        # Update token importance for all tokens within "important" tags
        for tagObject in soup.find_all(list(TAG_IMPORTANCE.keys())):
            for token in tokenizer.tokenize(str(tagObject.string)):
                if token in postings:
                    postings[token]["tokenImportance"] = TAG_IMPORTANCE[tagObject.name]

        return postings
    

    def writePartialIndexToDisk(self) -> None:
        """
        Saves the current inverted index to a partial index file which will later be merged
        with other partial index files to create a single index file.
        
        Each line contains a JOSN object containing a single key/value pair from the 
        invertedIndex. The tokens are sorted in ascending order to optimize merging step.
        """
     
        partialIndexFilePath = self.indexFolderPath / f"{PARTIAL_INDEX_FILE_NAME_PREFIX}_{self.currPartialIndex}.jsonl"

        with partialIndexFilePath.open("w", encoding="utf-8") as f:
            # Save tokens to file in sorted order to speed merging process
            for token in sorted(self.index):
                line = {token: self.index[token]}
                f.write(f"{json.dumps(line)}\n")

        if DEBUG: print(green(f"Successfully created: {partialIndexFilePath.name}\nLocation: {partialIndexFilePath.resolve()}"))
        
        # Keeps track of how many partial indexs have been created
        self.currPartialIndex += 1
   

    def mergePartialIndexesToDisk(self) -> None:
        """
        Merges all of the partial index files into a single index file called INDEX_FILE. 
        The merge preserves the alphabetical ordering of tokens.

        Examples:
            Case 1:

                partialIndex_1.txt    partialIndex_2.txt    partialIndex_3.txt    partialIndex_4.txt
                         \\                   //                    \\                    //
                            partialIndex_5.txt                          partialIndex_6.txt
                                        \\                                    //
                                                      INDEX_FILE

            Case 2:

                partialIndex_1.txt    partialIndex_2.txt    partialIndex_3.txt
                                \\            //                 //            
                                partialIndex_4.txt              //                        
                                        \\                     //
                                                INDEX_FILE 
        """

        if DEBUG: print(f"{' MERGING PARTIAL INDEXES ':=^{DEBUG_MESSAGE_WIDTH}}")

        while True:
            partialIndexPaths = [path for path in self.indexFolderPath.iterdir() if path.name.startswith(PARTIAL_INDEX_FILE_NAME_PREFIX)]

            if len(partialIndexPaths) == 1:
                if DEBUG: print(grey(f"RENAMING {partialIndexPaths[0].name} TO {INDEX_FILE_NAME}"))
                
                # All files have been merged, rename file to INDEX_FILE_NAME
                indexPath = partialIndexPaths[0].replace(self.indexFolderPath / INDEX_FILE_NAME)
                
                if DEBUG: print(green(f"Successfully created: {indexPath.name}\nLocation: {indexPath.resolve()}"))
                return

            
            partialIndexPaths = iter(partialIndexPaths)
            while True:
                try:
                    filePathA = next(partialIndexPaths)
                    filePathB = next(partialIndexPaths)
                    mergedFilePath = self.indexFolderPath / f"{PARTIAL_INDEX_FILE_NAME_PREFIX}_{self.currPartialIndex}.jsonl"
                    
                    if DEBUG: print(grey(f"MERGING {filePathA.name} and {filePathB.name} -> {mergedFilePath.name}"))
                    self._mergePartialIndexesToDisk(filePathA, filePathB, mergedFilePath)
                    self.currPartialIndex += 1

                    # Delete partial indexes since they are now merged
                    filePathA.unlink()
                    filePathB.unlink()

                except StopIteration:
                    break

        
    def _mergePartialIndexesToDisk(self, fileA: Path, fileB: Path, mergedFile: Path) -> None:
        fA = fileA.open(mode="r")
        fB = fileB.open(mode="r")

        with mergedFile.open(mode="w", encoding="utf-8") as fC:
            lineA = fA.readline().strip()
            lineB = fB.readline().strip()

            while (lineA != "") and (lineB != ""):
                dictA = json.loads(lineA)
                dictB = json.loads(lineB)

                tokenA = list(dictA.keys())[0]
                tokenB = list(dictB.keys())[0]

                if (tokenA == tokenB):
                    # If tokens are the same then merge objects and write to fC
                    merged = {tokenA: dictA[tokenA] + dictB[tokenB]}
                    fC.write(f"{json.dumps(merged)}\n")

                    lineA = fA.readline().strip() 
                    lineB = fB.readline().strip()

                # Maintian alphabetical order 
                elif (tokenA < tokenB):
                    fC.write(f"{lineA}\n")
                    lineA = fA.readline().strip()
                else:
                    fC.write(f"{lineB}\n")
                    lineB = fB.readline().strip()


            # Write rest of fA or fB to fC
            while (lineA != ""):
                fC.write(f"{lineA}\n")
                lineA = fA.readline().strip()
            while (lineB != ""):
                fC.write(f"{lineB}\n")
                lineB = fB.readline().strip()

        fA.close()
        fB.close()


    def calculateTF_IDF(self) -> None:
        """
        Iterates through the merged index and calculates the tf_idf score
        for every posting. Writes updated index back to disk.
        """

        if DEBUG: print(f"{' CALCULATING TD-IDF SCORES ':=^{DEBUG_MESSAGE_WIDTH}}")

        oldIndex = self.indexFolderPath / INDEX_FILE_NAME
        updatedIndex = self.indexFolderPath / "temp.txt"

        with oldIndex.open(mode="r", encoding="utf-8") as old, updatedIndex.open(mode="w", encoding="utf-8") as updated:
            for line in old:
                object = json.loads(line.strip())
                token = list(object.keys())[0]
                docFrequency = len(object[token])

                for posting in object[token]:
                    weight = posting.pop("tokenImportance", DEFAULT_IMPORTANCE)
                    log_tf = 1 + log10(posting["tokenFrequency"])
                    idf = log10(self.numberOfDocsIndexed / docFrequency)

                    posting["tf_idf"] = weight * log_tf * idf

                    # No longer need once tf.idf is calcuated
                    del posting["tokenFrequency"]
                    
                updated.write(f"{json.dumps(object)}\n")

        # Delete old index file
        oldIndex.unlink()

        # Rename updated index from 'temp.txt' to INDEX_FILE_NAME
        updatedIndex.replace(self.indexFolderPath / INDEX_FILE_NAME)

        if DEBUG: print(green(f"Successfully updated: {INDEX_FILE_NAME}"))


    def createIndexofIndex(self):
        """
        Indexes the start position of every token in index and saves it to the file INDEX_OF_INDEX_FILE_NAME. 
        This allows the search engine to run is cases where the index is too large to be loaded into memory.
        Instead the smaller index of index is loaded into memory.
        """

        if DEBUG: print(f"{' CREATING INDEX OF INDEX ':=^{DEBUG_MESSAGE_WIDTH}}")

        indexFilePath = self.indexFolderPath / INDEX_FILE_NAME

        indexOfIndex = dict()
        with indexFilePath.open(mode="r", encoding="utf-8") as indexFile:
            while True:
                # Get start posotion of current line
                seekPosition = indexFile.tell()
                line = indexFile.readline().strip()

                if line == "":
                    break

                object = json.loads(line.strip())
                token = list(object.keys())[0]

                indexOfIndex[token] = seekPosition


        indexOfIndexFilePath = self.indexFolderPath / INDEX_OF_INDEX_FILE_NAME

        with indexOfIndexFilePath.open(mode="w", encoding="utf-8") as indexOfIndexFile:
            json.dump(indexOfIndex, indexOfIndexFile, indent=2)

        if DEBUG: print(green(f"Successfully created: {indexOfIndexFilePath.name}\nLocation: {indexOfIndexFilePath.resolve()}"))

    
    def writeDocIdToURLToDisk(self) -> None:
        """
        Saves mapping of docID's -> URL to disk. Line N contains the URL for docID N-1. 
        (Ex. Line 10 contains URL for docID 9)
        """

        if DEBUG: print(f"{' SAVING docIDtoURL MAPPING TO DISK ':=^{DEBUG_MESSAGE_WIDTH}}")
        
        # Slash operator combines paths and creates new Path object to docIDtoURL file  
        docIDtoURLFilePath = self.indexFolderPath / DOCID_TO_URL_FILE_NAME

        with docIDtoURLFilePath.open(mode="w", encoding="utf-8") as f:
            for url in self.docIDtoURL:
                f.write(f"{url}\n")

        if DEBUG: print(green(f"Successfully created: {docIDtoURLFilePath.name}\nLocation {docIDtoURLFilePath.resolve()}"))


def main():
    with open(CONFIG_FILE_NAME) as f:
        cfg = json.load(f)
        corpusFolderPath = Path(cfg["CORPUS_PATH"])   # Directory containing JSON documents to parse
        indexFolderPath = Path(cfg["INDEX_STORAGE"])  # Directory to store index files

    
    if not corpusFolderPath.exists():
        print(red(f"Error: '{corpusFolderPath.resolve()}' does not exist"))
        sys.exit(1)
    
    if not indexFolderPath.exists():
        print(yellow(f"Warning: '{indexFolderPath.resolve()}' does not exist"))
        print(yellow(f"Would you like to create this folder (y/n)?"))
        ans = input().strip().lower()
        if ans == "y":
            indexFolderPath.mkdir()
        else:
            print(red(f"Error: Please set 'INDEX_STORAGE' to valid path in {CONFIG_FILE_NAME}"))
            sys.exit(1)


    Indexer(corpusFolderPath=corpusFolderPath, 
            indexFolderPath=indexFolderPath).run()


if __name__ == "__main__":
    main()