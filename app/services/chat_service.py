import json
from typing import Generator
import time

from app.services.session_service import SessionService
from core.ollama_service import OllamaService, OLLAMA_TOOLS, TOOL_REGISTRY
from core.prompts import SYSTEM_PROMPT
from concurrent.futures import ThreadPoolExecutor
from core.benchmark_metrics import save_benchmark, elapsed_ms


class ChatService:
    def __init__(self):
        self.ollama = OllamaService()
        self.session_service = SessionService()

    def stream_chat(self,
                    user_message: str,
                    session_id: str,
                    selected_model: str,
                    image_base64=None
                    ) -> Generator[str, None, None]:
        
        """
        Orchestrates the chat flow: 
        1. Retrieves history 
        2. Agentic Loop:
           a. Call LLM
           b. Stream content to user
           c. If tool calls requested -> Execute in parallel
           d. Append results to history and repeat
        3. Final response saving
        """
        request_start = time.perf_counter()
        first_response_token = None

        model_calls = []
        tool_calls_metrics = []
        iterations = 0
        max_iterations = 5  # Prevent infinite loops

        full_response_chunks = []

        # model_start = time.perf_counter()
        # first_model_token = None
        #
        # model_metrics = {
        #     "model_ttft_ms": None,
        #     "total_duration_ms": None,
        #     "load_duration_ms": None,
        #     "prompt_eval_duration_ms": None,
        #     "generation_duration_ms": None,
        #     "prompt_eval_count": None,
        #     "generation_token_count": None,
        #     "tokens_per_second": None,
        # }

        try:
            # 1. Initial Context Setup
            history_messages = self.session_service.get_session_history(session_id)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history_messages
            self.session_service.save_message(session_id, "user", user_message)
            
            if image_base64:
                messages.append({
                    "role": "user", 
                    "content": user_message, 
                    "images": [image_base64]
                })
            else:
                messages.append({
                    "role": "user", 
                    "content": user_message
                })

            # START AGENTIC LOOP
            while iterations < max_iterations:
                iterations += 1

                model_start = time.perf_counter()
                first_model_token = None

                model_metrics = {
                    "model_ttft_ms": None,
                    "total_duration_ms": None,
                    "load_duration_ms": None,
                    "prompt_eval_duration_ms": None,
                    "generation_duration_ms": None,
                    "prompt_eval_count": None,
                    "generation_token_count": None,
                    "tokens_per_second": None,
                }

                # A. Call LLM with current context and tools
                stream = self.ollama.chat(selected_model, messages, tools=OLLAMA_TOOLS)
                
                tool_calls_to_execute = []
                current_assistant_content = ""

                # B. Process the stream
                for chunk in stream:
                    message = chunk.get("message", {})
                    
                    # Capture any content the AI wants to say before/during tool calls
                    content = message.get("content", "")
                    if content:
                        now = time.perf_counter()

                        if first_model_token is None:
                            first_model_token = time.perf_counter()

                            model_metrics["model_ttft_ms"] = elapsed_ms(
                                model_start, now
                            )

                        if first_response_token is None:
                            first_response_token = now

                        current_assistant_content += content
                        full_response_chunks.append(content) # Optimized append
                        yield content

                    if chunk.get("done"):
                        model_metrics["total_duration_ms"] = (
                                chunk.get("total_duration", 0) / 1_000_000
                        )

                        model_metrics["load_duration_ms"] = (
                                chunk.get("load_duration", 0) / 1_000_000
                        )

                        model_metrics["prompt_eval_duration_ms"] = (
                                chunk.get("prompt_eval_duration", 0) / 1_000_000
                        )

                        model_metrics["generation_duration_ms"] = (
                                chunk.get("eval_duration", 0) / 1_000_000
                        )

                        model_metrics["prompt_eval_count"] = (
                            chunk.get("prompt_eval_count", 0)
                        )

                        model_metrics["generation_token_count"] = (
                            chunk.get("eval_count", 0)
                        )

                        duration = chunk.get("eval_duration", 0)

                        if duration > 0:
                            model_metrics["tokens_per_second"] = round(
                                chunk.get("eval_count", 0) /
                                (duration / 1_000_000_000),
                                2
                            )

                        model_calls.append(model_metrics.copy())

                    # Capture tool calls
                    if "tool_calls" in message and message["tool_calls"]:
                        tool_calls_to_execute.extend(message["tool_calls"])

                # C. Decision Point: Does the AI need tools?
                if not tool_calls_to_execute:
                    # No tools needed, the AI has provided the final answer
                    break 

                # D. Tool Execution Phase
                messages.append({
                    "role": "assistant",
                    "content": current_assistant_content,
                    "tool_calls": tool_calls_to_execute
                })

                def run_tool(tool_call):
                    func_info = tool_call.get("function", {})
                    func_name = func_info.get("name")
                    func_args = func_info.get("arguments", {})

                    tool_start = time.perf_counter()

                    try:
                        if func_name in TOOL_REGISTRY:
                            result = TOOL_REGISTRY[func_name](**func_args)
                        else:
                            result = f"Tool {func_name} not registered."

                    except Exception as e:
                        result = f"Error executing tool {func_name}: {str(e)}"

                    duration = elapsed_ms(tool_start)

                    return {
                        "name": func_name,
                        "result": result,
                        "duration_ms": duration
                    }

                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(run_tool, tool_calls_to_execute))

                for res in results:
                    tool_calls_metrics.append({
                        "name": res["name"],
                        "duration_ms": res["duration_ms"]
                    })

                    tool_msg = {
                        "role": "tool",
                        "content": (
                            json.dumps(res["result"])
                            if isinstance(res["result"], (dict, list))
                            else str(res["result"])
                        ),
                        "name": res["name"]
                    }

                    messages.append(tool_msg)
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield error_msg
            full_response_chunks.append(error_msg)


        finally:
            final_text = "".join(full_response_chunks)

            if final_text.strip():
                self.session_service.save_message(
                    session_id, "assistant", final_text
                )

            try:
                finished_at = time.perf_counter()

                save_benchmark({
                    "timestamp": time.time(),
                    "session_id": session_id,
                    "model": selected_model,
                    "iterations": iterations,
                    "total_latency_ms": elapsed_ms(
                        request_start, finished_at
                    ),
                    "end_to_end_ttft_ms": (
                        elapsed_ms(request_start, first_response_token)
                        if first_response_token is not None
                        else None
                    ),
                    "model_calls": model_calls,
                    "tool_calls": tool_calls_metrics
                })

            except Exception:
                import logging
                logging.exception("Error while saving benchmark")