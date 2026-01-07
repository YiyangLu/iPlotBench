"""Local LLM runner for iPlotBench evaluation.

Supports vLLM and other OpenAI-compatible endpoints with vision and tool calling.
Uses the same MCP Plotly tools as the Claude agent.
"""

import base64
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

from .mcp import (
    PLOTLY_TOOLS,
    execute_tool,
    get_session_store,
    register_session,
    unregister_session,
    session_context,
)


# Configuration presets matching original agent configs
AGENT_CONFIGS = {
    # vision: No MCP tools, direct image → JSON
    "vision": {
        "use_tools": False,
        "allowed_tools": [],
    },
    # vision_introspect: show_plot + get_plot_image + get_plot_json
    "vision_introspect": {
        "use_tools": True,
        "allowed_tools": ["show_plot", "get_plot_image", "get_plot_json"],
    },
    # vision_interactive: show_plot + get_plot_image + query_interactions + relayout + legendclick + selected
    "vision_interactive": {
        "use_tools": True,
        "allowed_tools": ["show_plot", "get_plot_image", "query_interactions", "relayout", "legendclick", "selected"],
    },
    # vision_introspect_interactive: All MCP tools
    "vision_introspect_interactive": {
        "use_tools": True,
        "allowed_tools": None,  # None = all tools
    },
}


@dataclass
class RunnerConfig:
    """Configuration for LocalLLMRunner."""

    api_base: str = "http://127.0.0.1:8666/v1"
    model: str = "Qwen/Qwen3-VL-4B-Instruct"
    api_key: str = "none"
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: int = 120
    use_tools: bool = True  # Whether to use function calling
    max_tool_rounds: int = 5  # Max tool call iterations
    logs_root: Path = Path("localenv/logs")  # Session logs directory
    allowed_tools: list[str] | None = None  # None = all tools, [] = no tools
    extra_headers: dict = field(default_factory=dict)  # Extra headers (e.g., for OpenRouter)

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "RunnerConfig":
        """Create config from preset name.

        Args:
            preset: "vision" or "vision_introspect_interactive"
            **overrides: Override any config fields
        """
        if preset not in AGENT_CONFIGS:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(AGENT_CONFIGS.keys())}")

        config_dict = AGENT_CONFIGS[preset].copy()
        config_dict.update(overrides)
        return cls(**config_dict)


@dataclass
class TaskResult:
    """Result from a task execution."""

    success: bool
    output: Optional[dict] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
    tool_calls: list = field(default_factory=list)


class LocalLLMRunner:
    """Run iPlotBench evaluation with local LLM via OpenAI-compatible API.

    Uses MCP Plotly tools for chart recreation and analysis.
    """

    def __init__(self, config: Optional[RunnerConfig] = None):
        self.config = config or RunnerConfig()
        self.session_messages: list[dict] = []
        self.session_id: Optional[str] = None
        self.session_store = get_session_store(self.config.logs_root)
        self._current_image_b64: Optional[str] = None

    def _encode_image(self, image_path: str | Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _get_allowed_tools(self) -> list[dict]:
        """Get filtered list of tools based on config."""
        if self.config.allowed_tools is None:
            # None = all tools
            return PLOTLY_TOOLS
        elif len(self.config.allowed_tools) == 0:
            # Empty list = no tools
            return []
        else:
            # Filter to allowed tools
            return [
                t for t in PLOTLY_TOOLS
                if t["function"]["name"] in self.config.allowed_tools
            ]

    def _call_api(
        self,
        messages: list[dict],
        use_tools: bool = False,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Call the LLM API."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if use_tools and self.config.use_tools:
            tools = self._get_allowed_tools()
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            **self.config.extra_headers,
        }
        response = requests.post(
            f"{self.config.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.config.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")

        return response.json()

    def _extract_json_from_response(self, content: str) -> Optional[dict]:
        """Extract JSON from response content."""
        # Try to find JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to parse entire content as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find any JSON object in content
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _extract_answer_from_response(self, content: str) -> Optional[int]:
        """Extract 0/1 answer from response."""
        content = content.strip()

        # Direct 0 or 1
        if content in ("0", "1"):
            return int(content)

        # Look for 0 or 1 in the response
        match = re.search(r"\b([01])\b", content)
        if match:
            return int(match.group(1))

        # Handle yes/no responses
        content_lower = content.lower()
        if any(w in content_lower for w in ["yes", "true", "correct"]):
            return 1
        if any(w in content_lower for w in ["no", "false", "incorrect"]):
            return 0

        return None

    def start_session(self, image_path: str | Path, figure_id: str) -> None:
        """Start a new evaluation session with an image.

        Args:
            image_path: Path to the input image
            figure_id: Figure identifier for session naming
        """
        # Create unique session ID
        self.session_id = f"{figure_id}_{uuid.uuid4().hex[:8]}"

        # Create session in store
        self.session_store.create_session(
            session_id=self.session_id,
            cwd=Path.cwd(),
            session_name=figure_id,
        )

        # Register session for tools
        register_session(self.session_store, self.session_id)

        # Encode image for embedding
        self._current_image_b64 = self._encode_image(image_path)

        # Start with empty message list (no system prompt, matching original)
        # Original Claude agent uses default system prompt
        self.session_messages = []

    def end_session(self) -> None:
        """End the current session."""
        if self.session_id:
            unregister_session()
            self.session_id = None

    def run_task1(self, image_path: str | Path, figure_id: str = "unknown") -> TaskResult:
        """Run Task 1: Recreation.

        Args:
            image_path: Path to input.png
            figure_id: Figure identifier

        Returns:
            TaskResult with Plotly figure JSON
        """
        # Start fresh session
        self.start_session(image_path, figure_id)

        # Task 1 prompt (exact match with original)
        # Original: "Read ./input.png and recreate this plot..."
        # For VLMs, we embed the image and use the same text
        prompt_text = """Read ./input.png and recreate this plot.

Output the Plotly figure as JSON with "data" and "layout" keys:
{"data": [...], "layout": {...}}"""

        # First message includes the image (for VLMs that need embedded images)
        self.session_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self._current_image_b64}"},
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        })

        try:
            # Try with tools first if enabled
            if self.config.use_tools:
                result = self._run_with_tools(task="task1")
                if result.success:
                    return result

            # Fallback to direct generation
            response = self._call_api(self.session_messages, use_tools=False)
            content = response["choices"][0]["message"]["content"]

            # Add to session
            self.session_messages.append({"role": "assistant", "content": content})

            # Extract JSON
            figure_json = self._extract_json_from_response(content)
            if figure_json and "data" in figure_json:
                return TaskResult(success=True, output=figure_json, raw_response=content)

            return TaskResult(
                success=False,
                error="Could not extract valid Plotly JSON",
                raw_response=content,
            )

        except Exception as e:
            return TaskResult(success=False, error=str(e))

    def run_task2(self, question: str) -> TaskResult:
        """Run Task 2: QA (within existing session).

        Args:
            question: The yes/no question to answer

        Returns:
            TaskResult with answer (0 or 1)
        """
        prompt = f"""{question}

Reply with ONLY a single digit: 0 or 1"""

        self.session_messages.append({"role": "user", "content": prompt})

        try:
            # Try with tools first if enabled
            if self.config.use_tools:
                result = self._run_with_tools(task="task2")
                if result.success:
                    return result

            # Fallback to direct generation
            response = self._call_api(
                self.session_messages, use_tools=False, max_tokens=64
            )
            content = response["choices"][0]["message"]["content"]

            # Add to session
            self.session_messages.append({"role": "assistant", "content": content})

            # Extract answer
            answer = self._extract_answer_from_response(content)
            if answer is not None:
                return TaskResult(
                    success=True, output={"answer": answer}, raw_response=content
                )

            return TaskResult(
                success=False,
                error="Could not extract 0/1 answer",
                raw_response=content,
            )

        except Exception as e:
            return TaskResult(success=False, error=str(e))

    def _run_with_tools(self, task: str) -> TaskResult:
        """Run task with tool calling support."""
        tool_calls_made = []

        for _ in range(self.config.max_tool_rounds):
            response = self._call_api(self.session_messages, use_tools=True)
            choice = response["choices"][0]
            message = choice["message"]

            # Check for tool calls
            if "tool_calls" in message and message["tool_calls"]:
                # Add assistant message with tool calls
                self.session_messages.append(message)

                # Track if we need to show images to the VLM
                images_to_show = []

                for tool_call in message["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}

                    # Execute tool
                    tool_result = execute_tool(tool_name, arguments)
                    tool_calls_made.append(
                        {"tool": tool_name, "args": arguments, "result": tool_result}
                    )

                    # For get_plot_image, extract base64 for VLM display
                    if tool_name == "get_plot_image" and tool_result.get("image_base64"):
                        images_to_show.append(tool_result["image_base64"])
                        # Remove base64 from tool response (too large for text)
                        tool_result_for_msg = {
                            "image_path": tool_result.get("image_path"),
                            "note": "Image displayed below for visual inspection."
                        }
                    else:
                        tool_result_for_msg = tool_result

                    # Add tool result to messages
                    self.session_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(tool_result_for_msg),
                        }
                    )

                    # Check if we got a final result from show_plot
                    if tool_name == "show_plot" and tool_result.get("success"):
                        # Get the plot JSON from the store
                        plot_id = tool_result.get("plot_id")
                        if plot_id and self.session_id:
                            fig_json = self.session_store.get_plot_json(
                                self.session_id, plot_id
                            )
                            if fig_json:
                                return TaskResult(
                                    success=True,
                                    output=fig_json,
                                    tool_calls=tool_calls_made,
                                )

                # If get_plot_image was called, add images as user message for VLM
                if images_to_show:
                    content = []
                    for img_b64 in images_to_show:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        })
                    content.append({
                        "type": "text",
                        "text": "Here is the plot image you requested. Please analyze it.",
                    })
                    self.session_messages.append({"role": "user", "content": content})
            else:
                # No tool calls, model gave direct response
                content = message.get("content", "")
                self.session_messages.append({"role": "assistant", "content": content})

                # Try to extract result from content
                if task == "task1":
                    figure_json = self._extract_json_from_response(content)
                    if figure_json and "data" in figure_json:
                        return TaskResult(
                            success=True,
                            output=figure_json,
                            raw_response=content,
                            tool_calls=tool_calls_made,
                        )
                else:  # task2
                    answer = self._extract_answer_from_response(content)
                    if answer is not None:
                        return TaskResult(
                            success=True,
                            output={"answer": answer},
                            raw_response=content,
                            tool_calls=tool_calls_made,
                        )

                break

        return TaskResult(
            success=False, error="Max tool rounds exceeded", tool_calls=tool_calls_made
        )


def run_figure(
    figure_id: str,
    config: RunnerConfig,
    data_dir: Path,
    output_dir: Path,
    config_name: str = "vision",
) -> dict[str, Any]:
    """Run evaluation for a single figure.

    Args:
        figure_id: Figure identifier (e.g., 'vbar_categorical_0000')
        config: Runner configuration
        data_dir: Path to env/ directory
        output_dir: Path to output directory
        config_name: Configuration name for output files

    Returns:
        Dict with task1 and task2 results
    """
    runner = LocalLLMRunner(config)

    test_dir = data_dir / "test" / figure_id
    query_dir = data_dir / "query" / figure_id
    figure_output_dir = output_dir / figure_id
    figure_output_dir.mkdir(parents=True, exist_ok=True)

    results = {"figure_id": figure_id, "task1": None, "task2": []}

    try:
        # Task 1: Recreation
        image_path = test_dir / "input.png"
        if image_path.exists():
            task1_result = runner.run_task1(image_path, figure_id)
            results["task1"] = {
                "success": task1_result.success,
                "error": task1_result.error,
            }

            # Save output
            task1_output_path = figure_output_dir / f"output_{config_name}_task1.json"
            if task1_result.success and task1_result.output:
                with open(task1_output_path, "w") as f:
                    json.dump(task1_result.output, f, indent=2)
            else:
                # Save empty/error output with raw response for debugging
                error_output = {"error": task1_result.error or "Unknown error", "data": [], "layout": {}}
                if task1_result.raw_response:
                    error_output["raw_response"] = task1_result.raw_response[:2000]  # Truncate for size
                with open(task1_output_path, "w") as f:
                    json.dump(error_output, f)

        # Task 2: QA
        questions_path = query_dir / "questions.json"
        if questions_path.exists():
            with open(questions_path) as f:
                questions = json.load(f)

            for i, q in enumerate(questions):
                question_str = q.get("question_string", q.get("question", ""))
                task2_result = runner.run_task2(question_str)

                results["task2"].append(
                    {
                        "question_idx": i,
                        "question_id": q.get("question_id", i),
                        "success": task2_result.success,
                        "answer": task2_result.output.get("answer") if task2_result.output else None,
                        "error": task2_result.error,
                    }
                )

                # Save output
                task2_output_path = figure_output_dir / f"output_{config_name}_task2_q{i}.json"
                if task2_result.success and task2_result.output:
                    with open(task2_output_path, "w") as f:
                        json.dump(task2_result.output, f)
                else:
                    with open(task2_output_path, "w") as f:
                        json.dump({"answer": None, "error": task2_result.error}, f)

    finally:
        # Always end session
        runner.end_session()

    return results
