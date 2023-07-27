from pathlib import Path
from functools import reduce
import tokenizer
import json
import time
import argparse


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
WEBAPP = False

# GLOBAL VARIABLES
invertedIndexFilePointer = None
indexOfIndex = {}
docIDtoURL = []




def loadIndexOfIndex() -> dict[str, int]:
    """
    Loads the index of index into memory.
    """

    with open(CONFIG_FILE) as f:
        # Gets path of folder containing the indexes
        indexFolderPath = json.load(f)["INDEX_STORAGE"]

    # Slash operator combines paths and creates new Path object to indexOfIndex file  
    indexOfIndexFilePath = Path(indexFolderPath) / "indexOfIndex.txt"
    
    indexOfIndex = dict()
    with indexOfIndexFilePath.open(mode="r", encoding="utf-8") as indexOfIndexFile:
        for line in indexOfIndexFile:
            token, seekPosition = line.strip().split()
            indexOfIndex[token] = int(seekPosition)

    return indexOfIndex





def loadURLs() -> list[str]:
    """
    Loads the URLs of entire corpus into memory. 
    The URL of a given docID is list[docID].
    """

    with open(CONFIG_FILE) as f:
        # Gets path of folder containing the indexes
        indexFolderPath = json.load(f)["INDEX_STORAGE"]

    # Slash operator combines paths and creates new Path object to docIDtoURL file  
    docIDtoURL_filePath = Path(indexFolderPath) / "docIDtoURL.txt"
    
    urlMap = []
    with docIDtoURL_filePath.open(mode="r", encoding="utf-8") as docIDtoURL_file:
        for line in docIDtoURL_file:
            urlMap.append(line.strip())

    return urlMap





def getPostingsListsIntersection(queryPostingsLists: list[ list[ dict[str, int] ] ]) -> list[int]:
    """
    Merges every query term posting list into a single list. 
    Resulting list contains the postings of docID's that contained
    every query term.
    """
    
    if len(queryPostingsLists) == 0:
        return [] 
    
    def intersect(p1List: list[dict[str, int]], p2List: list[dict[str, int]]):
        """
        Return the intersection of p1List and p2List. Intersection
        checks that twos postings contain the same docID. Also combines
        'tokenFrequency', 'tokenImportance', and 'tf_idf'
        """
        
        intersection = []
        p1_iter, p2_iter = iter(p1List), iter(p2List)
        
        try:
            p1, p2 = next(p1_iter), next(p2_iter)
            
            while True:
                if p1["docID"] == p2["docID"]:
                    intersection.append({
                        "docID" :           p1["docID"],
                        "tokenFrequency":   p1["tokenFrequency"]  + p2["tokenFrequency"],
                        "tokenImportance" : p1["tokenImportance"] + p2["tokenImportance"],
                        "tf_idf" :          p1["tf_idf"]          + p2["tf_idf"]
                    })
                    
                    p1, p2 = next(p1_iter), next(p2_iter)
                
                elif p1["docID"] < p2["docID"]:
                    p1 = next(p1_iter)   
                else:
                    p2 = next(p2_iter)

        except StopIteration:
            return intersection
    
    
    return list(reduce(intersect, queryPostingsLists))
    
    



def readPostingList(token: str) -> list[dict[str, int]]:
    """
    Returns the posting list of the specified token.
    """
    global invertedIndexFilePointer

    # Seek file pointer to line containing current token
    seekPostion = indexOfIndex.get(token, -1)
    if seekPostion == -1:
        return list()

    # Seek to line in index containing toker       
    invertedIndexFilePointer.seek(seekPostion)

    # Load JSON object into memery from invertedIndex
    json_dict = json.loads(invertedIndexFilePointer.readline().strip())

    return json_dict[token]["postingList"]










def getTop5Results(query: str) -> list[str]:
    global docIDtoURL

    stemmedQueryTokens = set(tokenizer.tokenize(query))

    queryPostingsLists = []
    for token in stemmedQueryTokens:
        queryPostingsLists.append(readPostingList(token))
 
    postingsIntersection = getPostingsListsIntersection(queryPostingsLists)
    postingsIntersection.sort(key=lambda x: x["tf_idf"] + x["tokenImportance"], reverse=True)
    
    urls = []
    try:
        for i in range(5):
            urls.append(docIDtoURL[postingsIntersection[i]["docID"]])
    except IndexError:
        pass
    finally:
        return urls



def runConsoleSearchEngine():
    """
    Main loop which continuously takes query input from 
    user until an empty query is given.
    """
    global indexOfIndex

    while (query := input("Input Query: ")) != "":
        start = time.time()
        urls = getTop5Results(query)
        end = time.time()
        
        print(f"(Search Time: {end - start:.4f} seconds)")
        for rank, url in enumerate(urls, 1):
            print(f"{rank}: {url}")
        




def runWebAppSearchEngine():
    from flask import Flask, render_template, request, url_for

    app = Flask(__name__)

    @app.route("/search")
    def result():
        query = request.args.to_dict().get("q", "")

        if query != "":
            start = time.time()
            urls = getTop5Results(query)
            end = time.time()

            search_time_milliseconds = round((end - start) * 1000, 4) 
            return render_template("search.html", query=query, urls=urls, search_time_milliseconds=search_time_milliseconds)
        
        return render_template("search.html")
    
    app.run(debug=True)





def parseCommandLineArgs():
    global WEBAPP

    parser = argparse.ArgumentParser()
    parser.add_argument("--webapp", action="store_true")

    args = parser.parse_args()

    if args.webapp:
        WEBAPP = True





def main():
    global invertedIndexFilePointer 
    global indexOfIndex
    global docIDtoURL
    
    parseCommandLineArgs()

    with open(CONFIG_FILE) as f:
        # Gets path of folder containing the indexes
        indexFolderPath = json.load(f)["INDEX_STORAGE"]

    # Load data needed for search engine into memeory
    invertedIndexFilePointer = open((Path(indexFolderPath) / "index.txt"), mode="r", encoding="utf-8")
    indexOfIndex             = loadIndexOfIndex()
    docIDtoURL               = loadURLs()


    if WEBAPP:
        # Runs webapp for entering queries
        runWebAppSearchEngine()
    else:
        # Runs the console UI for entering queries
        runConsoleSearchEngine()

    # Close the index file
    invertedIndexFilePointer.close()



if __name__ == "__main__":
    main()    