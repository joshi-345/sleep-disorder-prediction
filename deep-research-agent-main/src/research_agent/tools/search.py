from langchain.tools import tool
from langchain_tavily import TavilySearch

_search = TavilySearch(max_results=4)

@tool("web_search", description="Search the web for up-to-date information on a query.")
def web_search(query: str) -> str:
    return _search.invoke({"query": query})