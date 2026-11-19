import ollama
import httpx
from core.time_tool import tool as time_tool
from core.web_service import web_service

# Shared HTTP client with connection pooling
_ollama_http_client = httpx.Client(
    timeout=httpx.Timeout(300.0, connect=10.0),  # Long timeout for streaming
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    http2=True  # Ollama supports HTTP/2
)

# Pass the shared client to ollama
_ollama_client = ollama.Client(host="http://localhost:11434", timeout=300.0, follow_redirects=True)

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "time",
            "description": "Retrieves the current local system time (e.g. 10:15 PM). Use this whenever the user asks for the current time.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for real-time information or specific URLs",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query to look up"}},
                "required": ["query"],
            },
        }
    }
]

def web_search(query: str):
    search_results = web_service.search(query)
    if not search_results:
        return "No results found."

    # Use the new parallel scrape_multiple for Optimization #3 synergy
    urls = [r['url'] for r in search_results[:2] if r.get('url')]
    if urls:
        scraped_contents = web_service.scrape_multiple(urls)
        full_context = [
            f"Source: {url}\nContent: {content}" 
            for url, content in zip(urls, scraped_contents)
        ]
        return "\n\n---\n\n".join(full_context)
    
    return "No valid URLs to scrape."

TOOL_REGISTRY = {
    "time": time_tool.execute,
    "web_search": web_search,
}

class OllamaService:
    def __init__(self):
        # Use the shared client
        self.client = _ollama_client

    def list_models(self):
        try:
            models_info = self.client.list()
            model_names = [model["name"] for model in models_info.get("models", [])]
            return {"models": model_names}
        except Exception as e:
            return {"models": [], "error": str(e)}

    def chat(self, model: str, messages: list, tools: list | None = None):
        chat_args = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_ctx": 4096,
                "temperature": 0.5,
            }
        }
        if tools:
            chat_args["tools"] = tools
        return self.client.chat(**chat_args)

    def pull_model(self, model: str):
        return self.client.pull(model)

    def delete_model(self, model: str):
        return self.client.delete(model)

    def get_model_info(self, model: str):
        info = self.client.show(model)
        info_dict = info.model_dump() if hasattr(info, "model_dump") else dict(info)
        return {"model_name": model, "details": info_dict}