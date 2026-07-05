from pydantic import BaseModel,Field
from typing import Any,Dict,List

class Article(BaseModel):
    """A single PubMed article with metadata and evaluation results."""

    pubmed_id: int = Field(description="PubMed ID for this article.")
    doi: str = Field(description="The DOI string for this article.")
    title: str = Field(description="Article title")
    date: str = Field(description="Article publication date")
    authors: List[Dict[str, Any]] = Field(description="List of dicts, one per author. Dict must contain 'name' key.")
    source: str = Field(
        description="Article source, usually journal title. May be blank.",
        default=""
    )
    abstract: str = Field(
        description="Article abstract. May be blank if no abstract is available.",
        default=""
    )
    evaluation: str = Field(
        description="Text explaining whether this article is relevant to the search topic.",
        default=False
    )
    is_relevant: bool = Field(
        description="Whether or not this article is relevant to the search topic.",
        default=False
    )

class LitMonitorState(BaseModel):
    llm: str = Field(description="The LiteLLM path to the inference client we're using to talk to an LLM.")
    api_key : str = Field(
        description="Optional API key for inference client.",
        default="sk-placeholder",
        exclude=True
    )
    base_url: str = Field(description="URL where inference client is located.")
    sampling_params: Dict[str, Any] = Field(description="OpenAI-style dict of sampling parameters for the LLM.")
    topic_description: str = Field(description="A description of the topic the user is interested in.")
    search_terms: str = Field(description="The PubMed search terms being monitored")
    prior_pmids: List[str] = Field(
        default=[],
        description="A list of the PMIDs we have seen before."
    )
    new_articles: List[Article] = Field(
        default=[],
        description="A list of dicts containing each new article we have found."
    )
    num_pubmed_results: int = Field(
        default=25,
        description="The maximum number of search results to evaluate."
    )

    def get_article_with_pubmed_id(self, pmid:int) -> Article:
        """
        Get an article object with a given PubMed ID.

        Args:
            pmid (int): The PubMed ID to search for.
        
        Returns:
            The article, or None if no matching article was found.
        """
        for article in self.new_articles:
            if article.pubmed_id == pmid:
                return article
        # return none if not found
        return None