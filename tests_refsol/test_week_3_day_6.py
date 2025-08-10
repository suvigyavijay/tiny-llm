"""
Tests for Week 3, Day 6: AI Agent System with Tool Calling implementation.

Tests the ReAct agent system including tool registry, execution safety,
reasoning loops, and complete problem solving capabilities.
"""

import pytest
import mlx.core as mx
import json
import time
from unittest.mock import Mock, MagicMock, patch
from src.tiny_llm_ref.agent_system import (
    ToolCall, ToolResult, Tool, ToolRegistry, ReActAgent,
    ToolCallStatus, web_search_tool, calculator_tool, read_file_tool,
    write_file_tool, get_default_tools, create_agent_with_default_tools
)


class MockLLMModel:
    """Mock LLM for testing agent system."""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.prompts = []
    
    def generate(self, prompt):
        """Generate mock response based on predefined responses."""
        self.prompts.append(prompt)
        
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            # Default response
            response = "Thought: I need to solve this problem.\nFinal Answer: This is a mock response."
        
        self.call_count += 1
        return response


def mock_tool_function(x: int, y: int = 1) -> dict:
    """Mock tool function for testing."""
    return {"result": x + y, "operation": "addition"}


def failing_tool_function(should_fail: bool = True) -> dict:
    """Tool function that fails for testing error handling."""
    if should_fail:
        raise ValueError("This tool is designed to fail")
    return {"success": True}


class TestToolCall:
    """Test ToolCall data structure."""
    
    def test_tool_call_creation(self):
        """Test creating a tool call."""
        call = ToolCall(
            tool_name="test_tool",
            parameters={"param1": "value1", "param2": 42},
            call_id="call_123"
        )
        
        assert call.tool_name == "test_tool"
        assert call.parameters == {"param1": "value1", "param2": 42}
        assert call.call_id == "call_123"
    
    def test_tool_call_without_id(self):
        """Test tool call creation without explicit ID."""
        call = ToolCall(
            tool_name="test_tool",
            parameters={"param": "value"}
        )
        
        assert call.call_id is None


class TestToolResult:
    """Test ToolResult data structure."""
    
    def test_tool_result_success(self):
        """Test successful tool result."""
        result = ToolResult(
            call_id="call_123",
            status=ToolCallStatus.SUCCESS,
            result={"output": "success"},
            execution_time=0.5
        )
        
        assert result.call_id == "call_123"
        assert result.status == ToolCallStatus.SUCCESS
        assert result.result == {"output": "success"}
        assert result.error_message is None
        assert result.execution_time == 0.5
    
    def test_tool_result_error(self):
        """Test error tool result."""
        result = ToolResult(
            call_id="call_456",
            status=ToolCallStatus.ERROR,
            result=None,
            error_message="Tool execution failed",
            execution_time=0.1
        )
        
        assert result.status == ToolCallStatus.ERROR
        assert result.result is None
        assert result.error_message == "Tool execution failed"


class TestTool:
    """Test Tool definition."""
    
    def test_tool_creation(self):
        """Test creating a tool definition."""
        tool = Tool(
            name="test_tool",
            description="A tool for testing",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "First number"},
                    "y": {"type": "number", "description": "Second number"}
                },
                "required": ["x"]
            },
            function=mock_tool_function,
            safe=True,
            timeout=10.0
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "A tool for testing"
        assert tool.safe == True
        assert tool.timeout == 10.0
        assert tool.function == mock_tool_function


class TestToolRegistry:
    """Test ToolRegistry functionality."""
    
    def test_registry_initialization(self):
        """Test tool registry initialization."""
        registry = ToolRegistry()
        
        assert len(registry.tools) == 0
        assert len(registry.execution_history) == 0
    
    def test_register_tool(self):
        """Test tool registration."""
        registry = ToolRegistry()
        
        tool = Tool(
            name="test_tool",
            description="Test tool",
            parameters={"type": "object", "properties": {}},
            function=mock_tool_function
        )
        
        registry.register_tool(tool)
        
        assert "test_tool" in registry.tools
        assert registry.get_tool("test_tool") == tool
        assert "test_tool" in registry.list_tools()
    
    def test_get_tool_descriptions(self):
        """Test tool description formatting."""
        registry = ToolRegistry()
        
        tool = Tool(
            name="calculator",
            description="Performs calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"},
                    "precision": {"type": "number", "description": "Decimal precision"}
                },
                "required": ["expression"]
            },
            function=mock_tool_function
        )
        
        registry.register_tool(tool)
        descriptions = registry.get_tool_descriptions()
        
        assert "calculator" in descriptions
        assert "Performs calculations" in descriptions
        assert "expression" in descriptions
        assert "(required)" in descriptions
        assert "(optional)" in descriptions
    
    def test_execute_tool_success(self):
        """Test successful tool execution."""
        registry = ToolRegistry()
        
        tool = Tool(
            name="add_tool",
            description="Adds numbers",
            parameters={"type": "object", "properties": {}},
            function=mock_tool_function
        )
        registry.register_tool(tool)
        
        call = ToolCall(
            tool_name="add_tool",
            parameters={"x": 5, "y": 3},
            call_id="test_call"
        )
        
        result = registry.execute_tool(call)
        
        assert result.status == ToolCallStatus.SUCCESS
        assert result.result == {"result": 8, "operation": "addition"}
        assert result.error_message is None
        assert len(registry.execution_history) == 1
    
    def test_execute_tool_not_found(self):
        """Test execution of non-existent tool."""
        registry = ToolRegistry()
        
        call = ToolCall(
            tool_name="nonexistent_tool",
            parameters={},
            call_id="test_call"
        )
        
        result = registry.execute_tool(call)
        
        assert result.status == ToolCallStatus.ERROR
        assert "not found" in result.error_message
        assert result.result is None
    
    def test_execute_tool_parameter_validation(self):
        """Test tool execution with parameter validation."""
        registry = ToolRegistry()
        
        tool = Tool(
            name="test_tool",
            description="Test tool",
            parameters={
                "type": "object",
                "properties": {
                    "required_param": {"type": "string"}
                },
                "required": ["required_param"]
            },
            function=mock_tool_function
        )
        registry.register_tool(tool)
        
        # Missing required parameter
        call = ToolCall(
            tool_name="test_tool",
            parameters={},  # Missing required_param
            call_id="test_call"
        )
        
        result = registry.execute_tool(call)
        
        assert result.status == ToolCallStatus.ERROR
        assert "Missing required parameter" in result.error_message
    
    def test_execute_tool_error_handling(self):
        """Test tool execution error handling."""
        registry = ToolRegistry()
        
        tool = Tool(
            name="failing_tool",
            description="Tool that fails",
            parameters={"type": "object", "properties": {}},
            function=failing_tool_function
        )
        registry.register_tool(tool)
        
        call = ToolCall(
            tool_name="failing_tool",
            parameters={"should_fail": True},
            call_id="test_call"
        )
        
        result = registry.execute_tool(call)
        
        assert result.status == ToolCallStatus.ERROR
        assert "designed to fail" in result.error_message
        assert result.result is None
    
    def test_get_execution_stats(self):
        """Test execution statistics."""
        registry = ToolRegistry()
        
        # Register and execute tools
        tool = Tool("test_tool", "Test", {}, mock_tool_function)
        registry.register_tool(tool)
        
        # Execute successful calls
        for i in range(3):
            call = ToolCall("test_tool", {"x": i}, f"call_{i}")
            registry.execute_tool(call)
        
        # Execute failing call
        failing_tool = Tool("fail_tool", "Fail", {}, failing_tool_function)
        registry.register_tool(failing_tool)
        fail_call = ToolCall("fail_tool", {"should_fail": True}, "fail_call")
        registry.execute_tool(fail_call)
        
        stats = registry.get_execution_stats()
        
        assert stats["total_executions"] == 4
        assert stats["successful_executions"] == 3
        assert stats["success_rate"] == 0.75
        assert "average_execution_time" in stats
        assert "tool_usage" in stats


class TestReActAgent:
    """Test ReActAgent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        llm_model = MockLLMModel()
        tool_registry = ToolRegistry()
        
        agent = ReActAgent(
            llm_model=llm_model,
            tool_registry=tool_registry,
            max_iterations=5
        )
        
        assert agent.llm_model == llm_model
        assert agent.tool_registry == tool_registry
        assert agent.max_iterations == 5
        assert len(agent.conversation_history) == 0
    
    def test_parse_llm_response_complete(self):
        """Test parsing complete LLM response."""
        llm_model = MockLLMModel()
        tool_registry = ToolRegistry()
        agent = ReActAgent(llm_model, tool_registry)
        
        response = """
        Thought: I need to calculate something.
        Action: calculator
        Action Input: {"expression": "2 + 2"}
        Observation: The result is 4.
        Final Answer: The calculation result is 4.
        """
        
        parsed = agent.parse_llm_response(response)
        
        assert "I need to calculate something" in parsed["thought"]
        assert parsed["action"] == "calculator"
        assert parsed["action_input"] == {"expression": "2 + 2"}
        assert "The calculation result is 4" in parsed["final_answer"]
    
    def test_parse_llm_response_partial(self):
        """Test parsing partial LLM response."""
        llm_model = MockLLMModel()
        tool_registry = ToolRegistry()
        agent = ReActAgent(llm_model, tool_registry)
        
        response = """
        Thought: I'm thinking about this problem.
        Action: search_tool
        Action Input: search query here
        """
        
        parsed = agent.parse_llm_response(response)
        
        assert "thinking about this problem" in parsed["thought"]
        assert parsed["action"] == "search_tool"
        assert parsed["action_input"] == "search query here"  # String fallback
        assert parsed["final_answer"] is None
    
    def test_create_react_prompt(self):
        """Test ReAct prompt creation."""
        llm_model = MockLLMModel()
        tool_registry = ToolRegistry()
        
        # Add a tool
        tool = Tool("test_tool", "Test tool description", {}, mock_tool_function)
        tool_registry.register_tool(tool)
        
        agent = ReActAgent(llm_model, tool_registry)
        
        task = "Solve this problem"
        history = ["Previous: Some context"]
        
        prompt = agent.create_react_prompt(task, history)
        
        assert "Thought:" in prompt
        assert "Action:" in prompt
        assert "Action Input:" in prompt
        assert "test_tool" in prompt
        assert "Test tool description" in prompt
        assert "Solve this problem" in prompt
        assert "Some context" in prompt
    
    def test_step_execution(self):
        """Test single reasoning step execution."""
        # Mock response with tool call
        responses = [
            "Thought: I need to use a tool.\nAction: test_tool\nAction Input: {\"x\": 5, \"y\": 3}"
        ]
        llm_model = MockLLMModel(responses)
        tool_registry = ToolRegistry()
        
        # Add tool
        tool = Tool("test_tool", "Test tool", {}, mock_tool_function)
        tool_registry.register_tool(tool)
        
        agent = ReActAgent(llm_model, tool_registry)
        
        result = agent.step("Test problem")
        
        assert "parsed" in result
        assert "llm_response" in result
        assert "timestamp" in result
        
        parsed = result["parsed"]
        assert parsed["action"] == "test_tool"
        assert parsed["observation"] is not None
        assert "8" in parsed["observation"]  # 5 + 3 = 8
    
    def test_step_execution_no_action(self):
        """Test step execution without tool call."""
        responses = [
            "Thought: I can answer directly.\nFinal Answer: The answer is 42."
        ]
        llm_model = MockLLMModel(responses)
        tool_registry = ToolRegistry()
        
        agent = ReActAgent(llm_model, tool_registry)
        
        result = agent.step("Simple problem")
        
        parsed = result["parsed"]
        assert parsed["action"] is None
        assert parsed["observation"] is None
        assert "42" in parsed["final_answer"]
    
    def test_solve_problem_success(self):
        """Test successful problem solving."""
        responses = [
            "Thought: I need to calculate.\nAction: calculator\nAction Input: {\"expression\": \"10 + 5\"}",
            "Thought: The calculation shows 15.\nFinal Answer: The result is 15."
        ]
        llm_model = MockLLMModel(responses)
        tool_registry = ToolRegistry()
        
        # Add calculator tool
        calc_tool = Tool(
            "calculator",
            "Calculate expressions",
            {},
            lambda expression: {"result": 15}  # Mock calculation
        )
        tool_registry.register_tool(calc_tool)
        
        agent = ReActAgent(llm_model, tool_registry, max_iterations=3)
        
        solution = agent.solve("What is 10 + 5?")
        
        assert solution["success"] == True
        assert "15" in solution["solution"]
        assert solution["iterations"] <= 3
        assert len(solution["reasoning_steps"]) > 0
        assert len(solution["conversation_history"]) > 0
    
    def test_solve_problem_max_iterations(self):
        """Test problem solving hitting max iterations."""
        responses = [
            "Thought: Still thinking...",
            "Thought: Need more time...",
            "Thought: Almost there..."
        ]
        llm_model = MockLLMModel(responses)
        tool_registry = ToolRegistry()
        
        agent = ReActAgent(llm_model, tool_registry, max_iterations=2)
        
        solution = agent.solve("Complex problem")
        
        assert solution["success"] == False
        assert solution["iterations"] == 2
        assert "maximum number of iterations" in solution["solution"]
    
    def test_solve_problem_with_tool_error(self):
        """Test problem solving with tool errors."""
        responses = [
            "Thought: I'll use a tool.\nAction: failing_tool\nAction Input: {}",
            "Thought: The tool failed, but I can still answer.\nFinal Answer: Despite the error, the answer is known."
        ]
        llm_model = MockLLMModel(responses)
        tool_registry = ToolRegistry()
        
        # Add failing tool
        fail_tool = Tool("failing_tool", "Fails", {}, failing_tool_function)
        tool_registry.register_tool(fail_tool)
        
        agent = ReActAgent(llm_model, tool_registry)
        
        solution = agent.solve("Problem with tool error")
        
        assert solution["success"] == True
        assert "answer is known" in solution["solution"]
        # Should have error in conversation history
        history_str = "\n".join(solution["conversation_history"])
        assert "Error:" in history_str


class TestBuiltinTools:
    """Test built-in tool implementations."""
    
    def test_web_search_tool(self):
        """Test web search tool."""
        result = web_search_tool("machine learning", num_results=3)
        
        assert "query" in result
        assert "results" in result
        assert "total_results" in result
        
        assert result["query"] == "machine learning"
        assert len(result["results"]) == 3
        assert result["total_results"] == 3
        
        # Check result structure
        for search_result in result["results"]:
            assert "title" in search_result
            assert "url" in search_result
            assert "snippet" in search_result
    
    def test_calculator_tool_success(self):
        """Test calculator tool with valid expression."""
        result = calculator_tool("2 + 3 * 4")
        
        assert result["success"] == True
        assert "expression" in result
        assert "result" in result
        assert result["expression"] == "2 + 3 * 4"
        assert result["result"] == 14  # 2 + (3 * 4)
    
    def test_calculator_tool_invalid(self):
        """Test calculator tool with invalid expression."""
        result = calculator_tool("invalid expression with $ symbols")
        
        assert result["success"] == False
        assert "error" in result
        assert "invalid characters" in result["error"]
    
    @patch('builtins.open', create=True)
    def test_read_file_tool_success(self, mock_open):
        """Test file reading tool success."""
        mock_open.return_value.__enter__.return_value.read.return_value = "File content"
        
        result = read_file_tool("test.txt")
        
        assert result["success"] == True
        assert result["content"] == "File content"
        assert result["filepath"] == "test.txt"
        assert result["size"] == len("File content")
    
    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_read_file_tool_not_found(self, mock_open):
        """Test file reading tool with missing file."""
        result = read_file_tool("nonexistent.txt")
        
        assert result["success"] == False
        assert "error" in result
        assert "File not found" in result["error"]
    
    @patch('builtins.open', create=True)
    def test_write_file_tool_success(self, mock_open):
        """Test file writing tool success."""
        result = write_file_tool("output.txt", "Hello world")
        
        assert result["success"] == True
        assert result["filepath"] == "output.txt"
        assert "bytes_written" in result
    
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_write_file_tool_permission_error(self, mock_open):
        """Test file writing tool with permission error."""
        result = write_file_tool("readonly.txt", "content")
        
        assert result["success"] == False
        assert "error" in result
        assert "Permission denied" in result["error"]


class TestDefaultTools:
    """Test default tool configuration."""
    
    def test_get_default_tools(self):
        """Test default tool set."""
        tools = get_default_tools()
        
        assert len(tools) > 0
        assert all(isinstance(tool, Tool) for tool in tools)
        
        tool_names = [tool.name for tool in tools]
        assert "web_search" in tool_names
        assert "calculator" in tool_names
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        
        # Check safety flags
        for tool in tools:
            if tool.name in ["read_file", "write_file"]:
                assert tool.safe == False  # File operations are not safe
            else:
                assert tool.safe == True   # Other tools are safe
    
    def test_create_agent_with_default_tools(self):
        """Test agent creation with default tools."""
        llm_model = MockLLMModel()
        
        agent = create_agent_with_default_tools(llm_model)
        
        assert isinstance(agent, ReActAgent)
        assert agent.llm_model == llm_model
        assert len(agent.tool_registry.tools) > 0
        
        # Should have default tools available
        tool_names = agent.tool_registry.list_tools()
        assert "web_search" in tool_names
        assert "calculator" in tool_names


class TestAgentIntegration:
    """Integration tests for complete agent system."""
    
    def test_end_to_end_problem_solving(self):
        """Test complete end-to-end problem solving."""
        responses = [
            "Thought: I need to search for information.\nAction: web_search\nAction Input: {\"query\": \"Python programming\", \"num_results\": 2}",
            "Thought: Good, I found information about Python.\nAction: calculator\nAction Input: {\"expression\": \"2 + 2\"}",
            "Thought: Perfect! I have all the information needed.\nFinal Answer: Python is a programming language, and 2+2 equals 4."
        ]
        
        llm_model = MockLLMModel(responses)
        agent = create_agent_with_default_tools(llm_model)
        
        solution = agent.solve("Tell me about Python and calculate 2+2")
        
        assert solution["success"] == True
        assert "Python" in solution["solution"]
        assert "4" in solution["solution"]
        assert solution["iterations"] == 3
        
        # Check that tools were actually called
        stats = agent.tool_registry.get_execution_stats()
        assert stats["total_executions"] == 2  # web_search + calculator
        assert stats["successful_executions"] == 2
    
    def test_agent_with_complex_reasoning(self):
        """Test agent with complex multi-step reasoning."""
        responses = [
            "Thought: I need to gather information first.\nAction: web_search\nAction Input: {\"query\": \"machine learning basics\"}",
            "Thought: Now I need to calculate something.\nAction: calculator\nAction Input: {\"expression\": \"100 * 0.8\"}",
            "Thought: Let me search for more specific information.\nAction: web_search\nAction Input: {\"query\": \"neural networks\"}",
            "Thought: Perfect! Now I can provide a comprehensive answer.\nFinal Answer: Machine learning includes neural networks. With 80% accuracy on 100 samples, that's 80 correct predictions."
        ]
        
        llm_model = MockLLMModel(responses)
        agent = create_agent_with_default_tools(llm_model)
        
        solution = agent.solve("Explain machine learning and calculate 80% of 100")
        
        assert solution["success"] == True
        assert "neural networks" in solution["solution"]
        assert "80" in solution["solution"]
        
        # Verify reasoning steps
        assert len(solution["reasoning_steps"]) == 4
        assert len(solution["conversation_history"]) > 8  # Multiple thought/action/observation cycles
    
    def test_agent_error_recovery(self):
        """Test agent recovery from tool errors."""
        responses = [
            "Thought: I'll try to use a tool that might fail.\nAction: nonexistent_tool\nAction Input: {}",
            "Thought: That tool doesn't exist. Let me use a working tool.\nAction: calculator\nAction Input: {\"expression\": \"5 * 5\"}",
            "Thought: Great! The calculation worked.\nFinal Answer: Despite the initial error, I calculated 5*5 = 25."
        ]
        
        llm_model = MockLLMModel(responses)
        agent = create_agent_with_default_tools(llm_model)
        
        solution = agent.solve("Calculate 5 times 5")
        
        assert solution["success"] == True
        assert "25" in solution["solution"]
        
        # Should have error in conversation history but still succeed
        history_str = "\n".join(solution["conversation_history"])
        assert "Error:" in history_str
        assert "not found" in history_str
    
    def test_agent_performance_with_many_tools(self):
        """Test agent performance with many registered tools."""
        llm_model = MockLLMModel([
            "Thought: I'll solve this directly.\nFinal Answer: The answer is 42."
        ])
        
        agent = create_agent_with_default_tools(llm_model)
        
        # Register many additional tools
        for i in range(20):
            tool = Tool(
                f"tool_{i}",
                f"Description for tool {i}",
                {"type": "object", "properties": {}},
                lambda: {"result": f"output_{i}"}
            )
            agent.tool_registry.register_tool(tool)
        
        # Should still work efficiently
        start_time = time.time()
        solution = agent.solve("Simple problem")
        solve_time = time.time() - start_time
        
        assert solution["success"] == True
        assert solve_time < 5.0  # Should complete quickly
        
        # Tool descriptions should include all tools
        descriptions = agent.tool_registry.get_tool_descriptions()
        assert "tool_0" in descriptions
        assert "tool_19" in descriptions
    
    def test_agent_conversation_state(self):
        """Test agent conversation state management."""
        responses = [
            "Thought: Starting with first action.\nAction: calculator\nAction Input: {\"expression\": \"1 + 1\"}",
            "Thought: Now second calculation.\nAction: calculator\nAction Input: {\"expression\": \"2 * 3\"}",
            "Thought: Final step.\nFinal Answer: First result was 2, second was 6."
        ]
        
        llm_model = MockLLMModel(responses)
        agent = create_agent_with_default_tools(llm_model)
        
        solution = agent.solve("Do two calculations: 1+1 and 2*3")
        
        assert solution["success"] == True
        
        # Check conversation history preservation
        history = solution["conversation_history"]
        
        # Should contain both calculations
        history_str = "\n".join(history)
        assert "1 + 1" in history_str
        assert "2 * 3" in history_str
        
        # Should show progression through reasoning
        thought_lines = [line for line in history if line.startswith("Thought:")]
        assert len(thought_lines) == 3
        
        observation_lines = [line for line in history if line.startswith("Observation:")]
        assert len(observation_lines) == 2  # Two tool calls


if __name__ == "__main__":
    pytest.main([__file__])
