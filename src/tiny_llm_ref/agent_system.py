"""
AI Agent System with Tool Calling capabilities.

Implements a ReAct-style agent that can reason about problems and use external tools
to solve complex tasks through iterative reasoning and action.
"""

import mlx.core as mx
from typing import List, Dict, Optional, Any, Callable, Union
import json
import re
import time
import traceback
from dataclasses import dataclass
from enum import Enum


class ToolCallStatus(Enum):
    """Status of tool call execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class ToolCall:
    """Represents a tool call request."""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class ToolResult:
    """Result of a tool call execution."""
    call_id: str
    status: ToolCallStatus
    result: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class Tool:
    """Definition of an available tool."""
    name: str
    description: str
    parameters: Dict[str, Any]        # JSON schema for parameters
    function: Callable                # Actual function to execute
    safe: bool = True                 # Whether tool is safe to execute
    timeout: float = 30.0             # Execution timeout in seconds


class ToolRegistry:
    """Registry of available tools for the agent."""
    
    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Tool] = {}
        self.execution_history: List[ToolResult] = []
        
    def register_tool(self, tool: Tool):
        """
        Add tool to registry.
        
        Args:
            tool: Tool to register
        """
        self.tools[tool.name] = tool
        
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self.tools.get(tool_name)
        
    def list_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
        
    def get_tool_descriptions(self) -> str:
        """
        Format tools for LLM consumption.
        
        Returns:
            String description of all available tools
        """
        if not self.tools:
            return "No tools available."
            
        descriptions = ["Available tools:"]
        
        for tool in self.tools.values():
            desc = f"\n{tool.name}: {tool.description}"
            
            if tool.parameters:
                params = []
                for param_name, param_info in tool.parameters.get('properties', {}).items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    required = param_name in tool.parameters.get('required', [])
                    req_str = " (required)" if required else " (optional)"
                    params.append(f"  - {param_name} ({param_type}){req_str}: {param_desc}")
                    
                if params:
                    desc += "\n  Parameters:\n" + "\n".join(params)
            
            descriptions.append(desc)
            
        return "\n".join(descriptions)
        
    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Safely execute tool with error handling.
        
        Args:
            tool_call: Tool call to execute
            
        Returns:
            Tool execution result
        """
        start_time = time.time()
        call_id = tool_call.call_id or f"{tool_call.tool_name}_{int(time.time())}"
        
        # Check if tool exists
        tool = self.get_tool(tool_call.tool_name)
        if tool is None:
            return ToolResult(
                call_id=call_id,
                status=ToolCallStatus.ERROR,
                result=None,
                error_message=f"Tool '{tool_call.tool_name}' not found"
            )
        
        # Validate parameters
        validation_error = self._validate_parameters(tool, tool_call.parameters)
        if validation_error:
            return ToolResult(
                call_id=call_id,
                status=ToolCallStatus.ERROR,
                result=None,
                error_message=f"Parameter validation failed: {validation_error}"
            )
        
        # Execute tool with timeout and error handling
        try:
            # TODO: Implement proper timeout handling
            result = tool.function(**tool_call.parameters)
            execution_time = time.time() - start_time
            
            tool_result = ToolResult(
                call_id=call_id,
                status=ToolCallStatus.SUCCESS,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            tool_result = ToolResult(
                call_id=call_id,
                status=ToolCallStatus.ERROR,
                result=None,
                error_message=str(e),
                execution_time=execution_time
            )
        
        # Record execution
        self.execution_history.append(tool_result)
        
        return tool_result
        
    def _validate_parameters(self, tool: Tool, parameters: Dict[str, Any]) -> Optional[str]:
        """
        Validate tool parameters against schema.
        
        Args:
            tool: Tool definition
            parameters: Parameters to validate
            
        Returns:
            Error message if validation fails, None if valid
        """
        schema = tool.parameters
        
        # Check required parameters
        required = schema.get('required', [])
        for req_param in required:
            if req_param not in parameters:
                return f"Missing required parameter: {req_param}"
        
        # Check parameter types (simplified validation)
        properties = schema.get('properties', {})
        for param_name, param_value in parameters.items():
            if param_name in properties:
                expected_type = properties[param_name].get('type')
                if expected_type == 'string' and not isinstance(param_value, str):
                    return f"Parameter {param_name} should be string, got {type(param_value)}"
                elif expected_type == 'number' and not isinstance(param_value, (int, float)):
                    return f"Parameter {param_name} should be number, got {type(param_value)}"
                elif expected_type == 'boolean' and not isinstance(param_value, bool):
                    return f"Parameter {param_name} should be boolean, got {type(param_value)}"
        
        return None
        
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics."""
        if not self.execution_history:
            return {"total_executions": 0}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for result in self.execution_history 
                                  if result.status == ToolCallStatus.SUCCESS)
        avg_execution_time = sum(result.execution_time for result in self.execution_history) / total_executions
        
        # Tool usage counts
        tool_usage = {}
        for result in self.execution_history:
            tool_name = result.call_id.split('_')[0]  # Extract tool name from call_id
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions,
            "average_execution_time": avg_execution_time,
            "tool_usage": tool_usage
        }


class ReActAgent:
    """ReAct (Reasoning and Acting) agent implementation."""
    
    def __init__(self, llm_model: Any, tool_registry: ToolRegistry, max_iterations: int = 10):
        """
        Initialize ReAct agent.
        
        Args:
            llm_model: Language model for reasoning
            tool_registry: Registry of available tools
            max_iterations: Maximum reasoning iterations
        """
        self.llm_model = llm_model
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        
        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.current_task = None
        
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM output for thoughts, actions, and observations.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed response with thoughts, actions, etc.
        """
        parsed = {
            "thought": None,
            "action": None,
            "action_input": None,
            "observation": None,
            "final_answer": None
        }
        
        # Extract thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|$)", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            parsed["thought"] = thought_match.group(1).strip()
        
        # Extract action
        action_match = re.search(r"Action:\s*(.*?)(?=\n|$)", response, re.IGNORECASE)
        if action_match:
            parsed["action"] = action_match.group(1).strip()
        
        # Extract action input
        action_input_match = re.search(r"Action Input:\s*(.*?)(?=\n(?:Observation|Thought|Final Answer):|$)", 
                                     response, re.DOTALL | re.IGNORECASE)
        if action_input_match:
            action_input_str = action_input_match.group(1).strip()
            try:
                # Try to parse as JSON
                parsed["action_input"] = json.loads(action_input_str)
            except json.JSONDecodeError:
                # Fall back to string
                parsed["action_input"] = action_input_str
        
        # Extract final answer
        final_answer_match = re.search(r"Final Answer:\s*(.*?)$", response, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            parsed["final_answer"] = final_answer_match.group(1).strip()
        
        return parsed
        
    def create_react_prompt(self, task: str, conversation_history: List[str]) -> str:
        """
        Create ReAct prompt with task and available tools.
        
        Args:
            task: Current task to solve
            conversation_history: Previous conversation steps
            
        Returns:
            Complete prompt for LLM
        """
        tools_description = self.tool_registry.get_tool_descriptions()
        
        history_str = "\n".join(conversation_history) if conversation_history else "None"
        
        prompt = f"""You are a helpful assistant that can use tools to solve problems. Use the ReAct format:

Thought: Think about what you need to do
Action: Choose a tool to use
Action Input: Provide parameters for the tool (as JSON)
Observation: I will provide the tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Final Answer: Provide your final answer

{tools_description}

Previous conversation:
{history_str}

Task: {task}

Begin your reasoning:"""

        return prompt
        
    def step(self, user_message: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        """
        Execute one reasoning step.
        
        Args:
            user_message: User's request or task
            conversation_history: Previous conversation steps
            
        Returns:
            Step result with thoughts, actions, and observations
        """
        if conversation_history is None:
            conversation_history = []
        
        # Create prompt
        prompt = self.create_react_prompt(user_message, conversation_history)
        
        # Generate response
        llm_response = self.llm_model.generate(prompt)
        
        # Parse response
        parsed = self.parse_llm_response(llm_response)
        
        # Execute action if present
        if parsed["action"] and parsed["action_input"]:
            tool_call = ToolCall(
                tool_name=parsed["action"],
                parameters=parsed["action_input"] if isinstance(parsed["action_input"], dict) else {}
            )
            
            tool_result = self.tool_registry.execute_tool(tool_call)
            
            if tool_result.status == ToolCallStatus.SUCCESS:
                parsed["observation"] = str(tool_result.result)
            else:
                parsed["observation"] = f"Error: {tool_result.error_message}"
        
        return {
            "llm_response": llm_response,
            "parsed": parsed,
            "timestamp": time.time()
        }
        
    def solve(self, problem: str) -> Dict[str, Any]:
        """
        Solve a problem using iterative reasoning and tool usage.
        
        Args:
            problem: Problem description
            
        Returns:
            Solution with reasoning trace
        """
        self.current_task = problem
        conversation_history = []
        steps = []
        
        for iteration in range(self.max_iterations):
            # Execute reasoning step
            step_result = self.step(problem, conversation_history)
            steps.append(step_result)
            
            parsed = step_result["parsed"]
            
            # Add step to conversation history
            if parsed["thought"]:
                conversation_history.append(f"Thought: {parsed['thought']}")
            if parsed["action"]:
                conversation_history.append(f"Action: {parsed['action']}")
            if parsed["action_input"]:
                action_input_str = json.dumps(parsed["action_input"]) if isinstance(parsed["action_input"], dict) else str(parsed["action_input"])
                conversation_history.append(f"Action Input: {action_input_str}")
            if parsed["observation"]:
                conversation_history.append(f"Observation: {parsed['observation']}")
            
            # Check if we have a final answer
            if parsed["final_answer"]:
                return {
                    "solution": parsed["final_answer"],
                    "reasoning_steps": steps,
                    "iterations": iteration + 1,
                    "conversation_history": conversation_history,
                    "success": True
                }
        
        # Max iterations reached without final answer
        return {
            "solution": "Could not solve the problem within the maximum number of iterations.",
            "reasoning_steps": steps,
            "iterations": self.max_iterations,
            "conversation_history": conversation_history,
            "success": False
        }


# Built-in tool implementations

def web_search_tool(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Mock web search tool (in practice would use real search API).
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Search results
    """
    # Mock search results
    results = [
        {
            "title": f"Search result {i+1} for '{query}'",
            "url": f"https://example.com/result{i+1}",
            "snippet": f"This is a mock search result snippet for query '{query}'. Result number {i+1}."
        }
        for i in range(num_results)
    ]
    
    return {
        "query": query,
        "results": results,
        "total_results": num_results
    }


def calculator_tool(expression: str) -> Dict[str, Any]:
    """
    Safe calculator tool for mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Calculation result
    """
    try:
        # Simple safe evaluation (in practice would use a proper math parser)
        # Only allow basic arithmetic operations
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains invalid characters")
        
        # Evaluate safely
        result = eval(expression)
        
        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "expression": expression,
            "result": None,
            "error": str(e),
            "success": False
        }


def read_file_tool(filepath: str) -> Dict[str, Any]:
    """
    Read file contents safely.
    
    Args:
        filepath: Path to file to read
        
    Returns:
        File contents or error
    """
    try:
        # In practice would have proper sandboxing and validation
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "filepath": filepath,
            "content": content,
            "size": len(content),
            "success": True
        }
    except Exception as e:
        return {
            "filepath": filepath,
            "content": None,
            "error": str(e),
            "success": False
        }


def write_file_tool(filepath: str, content: str) -> Dict[str, Any]:
    """
    Write content to file safely.
    
    Args:
        filepath: Path to file to write
        content: Content to write
        
    Returns:
        Write result
    """
    try:
        # In practice would have proper sandboxing and validation
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "filepath": filepath,
            "bytes_written": len(content.encode('utf-8')),
            "success": True
        }
    except Exception as e:
        return {
            "filepath": filepath,
            "error": str(e),
            "success": False
        }


def get_default_tools() -> List[Tool]:
    """Get default set of tools for agents."""
    return [
        Tool(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "number", "description": "Number of results to return (default: 5)"}
                },
                "required": ["query"]
            },
            function=web_search_tool
        ),
        Tool(
            name="calculator",
            description="Calculate mathematical expressions",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                },
                "required": ["expression"]
            },
            function=calculator_tool
        ),
        Tool(
            name="read_file",
            description="Read contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to file to read"}
                },
                "required": ["filepath"]
            },
            function=read_file_tool,
            safe=False  # File access requires permission
        ),
        Tool(
            name="write_file",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to file to write"},
                    "content": {"type": "string", "description": "Content to write to file"}
                },
                "required": ["filepath", "content"]
            },
            function=write_file_tool,
            safe=False  # File access requires permission
        )
    ]


def create_agent_with_default_tools(llm_model: Any) -> ReActAgent:
    """
    Create agent with default tool set.
    
    Args:
        llm_model: Language model for the agent
        
    Returns:
        Configured ReAct agent
    """
    tool_registry = ToolRegistry()
    
    for tool in get_default_tools():
        tool_registry.register_tool(tool)
    
    return ReActAgent(llm_model, tool_registry)
