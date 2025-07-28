from pathlib import Path
import json, time, argparse

import tokenizer


# INVERTED INDEX JSON STRUCTURE
#
# {
#   "TOKEN": [
#     {
#       "docID": int,
#       "tf_idf": float
#     }, 
#   ]
# }
       


# GLOBAL VARIABLES
CONFIG_FILE = "config.json"


class Searcher:
    def __init__(self, indexFilePath: Path, metaIndexFilePath: Path, docIDtoURLFilePath: Path, runAsWebApp: bool = False):
        self.indexFilePath = indexFilePath
        self.metaIndexFilePath = metaIndexFilePath
        self.docIDtoURLFilePath = docIDtoURLFilePath
        self.runAsWebApp = runAsWebApp
        
        self.metaIndex = dict()
        self.docIDtoURL = []

    def run(self):
        self.indexFile = open(self.indexFilePath, mode="r", encoding="utf-8")
        self.loadMetaIndex()
        self.loadDocIDtoURLs()

        if self.runAsWebApp:
            self.runWebAppSearchEngine()
        else:
            self.runConsoleSearchEngine()

        self.indexFile.close()
        
    def loadMetaIndex(self) -> None:
        with self.metaIndexFilePath.open(mode="r", encoding="utf-8") as metaIndexFile:
            self.metaIndex = json.load(metaIndexFile)


    def loadDocIDtoURLs(self) -> None:
        """
        Loads the URLs of entire corpus into memory. 
        The URL of a given docID is list[docID].
        """
        
        with self.docIDtoURLFilePath.open(mode="r", encoding="utf-8") as docIDtoURLFile:
            self.docIDtoURL = json.load(docIDtoURLFile)
    
    
    def runWebAppSearchEngine(self):
        from flask import Flask, render_template, request

        app = Flask(__name__)

        @app.route("/")
        @app.route("/search")
        def result():
            query = request.args.to_dict().get("q", "")

            if query != "":
                start = time.time()
                urls = self.searchQuery(query)
                end = time.time()
                search_time_milliseconds = round((end - start) * 1000, 3) 
                
                return render_template("search.html", query=query, urls=urls, search_time_milliseconds=search_time_milliseconds)

            return render_template("search.html")

        app.run(debug=True)


    def runConsoleSearchEngine(self):
        """
        Continuously takes query input from user until an empty query is given.
        """

        try :
            while (query := input("Input Query: ")) != "":
                start = time.time()
                urls = self.searchQuery(query)
                end = time.time()

                print(f"(Search Time: {end - start:.4f} seconds)")
                for rank, url in enumerate(urls, 1):
                    print(f"{rank}: {url}")
        except EOFError:
            return


    def searchQuery(self, query: str, maxResults: int = 5) -> list[str]:
        """
        Returns the most relevant URLs for the given query. Defaults to 5 URLs.
        """
        
        queryTerms = set(tokenizer.tokenize(query))
        queryPostingsLists = [self.readPostingList(token) for token in queryTerms]
        
        documents = self.getPostingsListsIntersection(queryPostingsLists)
        
        # If true then no documents contained ALL query terms. 
        # Instead find all documents with at least ONE query term
        if len(documents) == 0:
            documents = self.mergePostingLists(queryPostingsLists)

        # Sort documents with highest tf.idf score first
        documents.sort(key=lambda x: x["tf_idf"], reverse=True)

        urls = []
        for index, posting in enumerate(documents):
            if index == maxResults:
                break

            docID = posting["docID"]
            urls.append(self.docIDtoURL[docID])

        return urls
        

    def readPostingList(self, token: str) -> list[dict[str, int]]:
        """
        Returns the posting list of the specified token from the index file.
        """

        # Seek file pointer to line containing current token
        seekPostion = self.metaIndex.get(token, -1)
        if seekPostion == -1:
            return list()

        # Seek to line in index containing toker       
        self.indexFile.seek(seekPostion)

        # Load JSON object into memory from invertedIndex
        json_dict = json.loads(self.indexFile.readline().strip())

        return json_dict[token]
        


    def getPostingsListsIntersection(self, queryPostingsLists: list[list[dict[str, int]]]) -> list[dict[str, int]]:
        """
        Finds all the postings with a given docID that appear in each list. The tf.idf scores of 
        each postings with the same docID is accumulated
        
        Ex:
            Input: [
                     [{docID: 1, tf_idf: 1.3}, {docID: 2, tf_idf: 1.5}],  # computer
                     [{docID: 2, tf_idf: 1.7}, {docID: 3, tf_idf: 2.3}]   # science
                   ]

            Output: [{docID: 2, tf_idf: 3.2}]
        """
            
        
        if len(queryPostingsLists) == 0:
            return [] 
        
        # Sort lists in ascending order by length to improve intersect performance
        queryPostingsLists.sort(key=len)

        res = queryPostingsLists[0]
        for i in range(1, len(queryPostingsLists)):
            res = self._intersect(res, queryPostingsLists[i])
            
        return res
    

    def _intersect(self, p1: list[dict[str, int]], p2: list[dict[str, int]]) -> list[dict[str, int]]:
        intersection = []
        i, j = 0, 0

        while i < len(p1) and j < len(p2):
            if p1[i]["docID"] == p2[j]["docID"]:
                intersection.append({
                            "docID": p1[i]["docID"],
                            "tf_idf": p1[i]["tf_idf"] + p2[j]["tf_idf"]
                    })
                i += 1
                j += 1
            elif p1[i]["docID"] < p2[j]["docID"]:
                i += 1
            else:
                j += 1

        return intersection
    
    
    def mergePostingLists(self, queryPostingsLists: list[list[dict[str, int]]]) -> list[dict[str, int]]:
        """
        Merges all posting lists into a single list summing up the tf.idf scores of postings with
        the same docID for each token.

        Ex:
            Input: [
                     [{docID: 1, tf_idf: 1.3}, {docID: 2, tf_idf: 1.5}],  # computer
                     [{docID: 2, tf_idf: 1.7}, {docID: 3, tf_idf: 2.3}]   # science
                   ]

            Output: [{docID: 1, tf_idf: 1.3}, {docID: 2, tf_idf: 3.2}, {docID: 3, tf_idf: 2.3}]
        """
            
        
        if len(queryPostingsLists) == 0:
            return [] 
        
        res = queryPostingsLists[0]
        for i in range(1, len(queryPostingsLists)):
            res = self._merge(res, queryPostingsLists[i])
            
        return res
    

    def _merge(self, p1: list[dict[str, int]], p2: list[dict[str, int]]) -> list[dict[str, int]]:
        merged = []
        i, j = 0, 0

        while i < len(p1) and j < len(p2):
            if p1[i]["docID"] == p2[j]["docID"]:
                merged.append({
                        "docID": p1[i]["docID"],
                        "tf_idf": p1[i]["tf_idf"] + p2[j]["tf_idf"]
                })
                i += 1
                j += 1
            elif p1[i]["docID"] < p2[j]["docID"]:
                i += 1
            else:
                j += 1

        while i < len(p1):
            merged.append(p1[i])
            i += 1
        while j < len(p2):
            merged.append(p2[j])
            j += 1
            
        return merged
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webapp", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
        
        indexFolderPath = Path(cfg["INDEX_STORAGE"])  # Directory containing index files

        indexFilePath      = indexFolderPath / cfg["INDEX_FILE"]
        metaIndexFilePath  = indexFolderPath / cfg["META_INDEX_FILE"]
        docIDtoURLFilePath = indexFolderPath / cfg["ID_TO_URL_FILE"]

    Searcher(indexFilePath=indexFilePath, 
             metaIndexFilePath=metaIndexFilePath,
             docIDtoURLFilePath=docIDtoURLFilePath,
             runAsWebApp=args.webapp).run()


if __name__ == "__main__":
    main()    