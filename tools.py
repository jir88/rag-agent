import requests
import time

from typing import TypedDict, List, Dict, Any, Optional
from xml.etree import ElementTree

class Tool:
    """
    Simple base tool class.
    """

    llm_definition:Dict
    """Tool description dict to be sent to the LLM."""

    def call(self, *args, **kwargs):
        """Needs to be implemented."""
        raise NotImplementedError("Subclass must implement the 'call' method.")

class PubmedSearchTool(Tool):
    """
    Searches PubMed and returns the top N articles matching a given query.
    """

    # definition that gets sent to the LLM
    llm_definition = {
        "type": "function",
        "function": {
            "name": "search_pubmed",
            "description": "Search the PubMed database for published scientific articles in biomedical-related fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A query string using PubMed query syntax. Multiple terms should be joined with AND or OR (e.g. stroke AND genomics)."
                    }
                },
                "required": ["query"],
            },
        },
    }

    def call(self, query:str, start_results:int=0, max_results:int=10, sort:str='pub_date') -> List[Dict]:
        """
        Searches PubMed for scientific publications using a given query. 
        Returns a list of dicts containing the metadata
        of the first N search results. The dict has keys 'pubmed_id', 'doi', title',
        'date', 'authors', 'source', and 'abstract'.

        Parameters:
            query (str): A query string using PubMed query syntax. Multiple terms should be joined with AND or OR (e.g. stroke AND genomics).
            start_results (int): The index of the first result to retrieve, zero-indexed; default 0.
            max_results (int): Maximum number of articles to retrieve, default 10.
            sort (str): How to sort search results. 'pub_date' returns most recently published results. 'relevance' returns the best matches to the query.

        Returns:
            List[Dict]: List of dictionaries containing article metadata and abstracts.
        """
        # Base URLs for PubMed E-Utilities
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        # Initial search to get IDs
        esearch_params = {
            "db": "pubmed",
            "term": query,
            "sort": sort,
            "retstart": start_results,
            "retmax": max_results,
            "usehistory": "y",
            "retmode": "json"
        }
        response = requests.get(esearch_url, params=esearch_params)
        response.raise_for_status()
        search_data = response.json()
        
        webenv = search_data["esearchresult"]["webenv"]
        query_key = search_data["esearchresult"]["querykey"]
        ids = search_data["esearchresult"]["idlist"]

        # if we didn't get any results, return empty list
        if len(ids) == 0:
            return []
        
        # Fetch summaries
        esummary_params = {
            "db": "pubmed",
            "query_key": query_key,
            "WebEnv": webenv,
            "retmode": "json",
            "retstart": start_results,
            "retmax": max_results,
        }
        summary_response = requests.get(esummary_url, params=esummary_params)
        summary_response.raise_for_status()
        summaries = summary_response.json()
        # Fetch abstracts in a batch
        efetch_params = {
            "db": "pubmed",
            "query_key": query_key,
            "WebEnv": webenv,
            #"id": ",".join(batch_ids),
            "retstart": start_results,
            "retmax": max_results,
            #"rettype": "abstract",
            #"retmode": "text"
            "retmode": "xml"
        }
        fetch_response = requests.get(efetch_url, params=efetch_params)
        fetch_response.raise_for_status()
        # returns dict mapping PMID->abstract
        abstracts = self.parse_abstracts(fetch_response.text)

        results = []
        # for each PMID returned, pull all data
        for pmid in ids:
            # get metadata
            article_meta = summaries['result'][pmid]
            results.append({
                    "pubmed_id": pmid,
                    "doi": article_meta.get('elocationid', ""),
                    "title": article_meta.get("title", ""),
                    "date": article_meta.get('sortpubdate', ""),
                    "authors": article_meta.get('authors', []),
                    "source": article_meta.get('source', ""),
                    "abstract": abstracts.get(pmid, "")
                })
        # return list of articles
        return results

    def parse_abstracts(self, xml_response:str):
        """
        Parse abstracts from PubMed XML response.

        Parameters:
            xml_response (str): XML response from PubMed.

        Returns:
            dict: Dictionary mapping article IDs to their abstracts.
        """
        root = ElementTree.fromstring(xml_response)
        abstracts = {}
        for article in root.findall(".//PubmedArticle"):
            pmid = article.find(".//PMID").text
            abstract_text = ""
            # abstracts are made out of pieces of text
            for abstract in article.findall(".//AbstractText"):
                # sometimes these have labels, like "METHODS"
                label = abstract.attrib.get('Label')
                if label is not None:
                    abstract_text += "\n\n" + label + "\n\n"
                # now add the main body of the element
                if abstract.text is not None:
                    abstract_text += abstract.text + " "
                # sometimes formatting (esp. super/subscripting) is
                # done with XML tags, so we have to fight those
                for part in abstract:
                    if part.text:
                        abstract_text += part.text
                    if part.tail:
                        abstract_text += part.tail
            abstracts[pmid] = abstract_text.strip()
        return abstracts