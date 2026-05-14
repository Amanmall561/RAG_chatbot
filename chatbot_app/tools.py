import datetime
import numexpr as ne
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from .rag import retrieve_context

# Web Search Tool
web_search_run = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    try:
        return web_search_run.run(query)
    except Exception as e:
        return f"Error performing web search: {str(e)}"

@tool
def document_search(query: str) -> str:
    return retrieve_context(query)

@tool
def calculator(expression: str) -> str:
    try:
        # evaluate safely with numexpr
        result = ne.evaluate(expression)
        return str(result.item() if hasattr(result, "item") else result)
    except Exception as e:
        return f"Error calculating expression '{expression}': {str(e)}"

@tool
def get_current_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

AGENT_TOOLS = [document_search, web_search, calculator, get_current_time]
