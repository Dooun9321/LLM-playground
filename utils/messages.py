"""
LangChain/LangGraph 메시지 스트리밍 및 처리 유틸리티 모듈

최신 LangChain v1.0+ 및 LangGraph 패턴을 지원합니다.

Note:
    - LangChain v1.0에서 ToolAgentAction, AgentAction, AgentStep 등은 deprecated됨
    - 새로운 에이전트는 create_agent 및 tool_calls 패턴을 사용
"""
from langchain_core.messages import AIMessageChunk
from typing import Any, Dict, List, Callable, Optional, Union, Literal, AsyncIterator
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
import uuid


# ============================================================
# 상수 정의
# ============================================================
SEPARATOR_WIDTH = 50
SEPARATOR_LINE = "=" * SEPARATOR_WIDTH
SEPARATOR_DASH = "- " * 25

# ANSI 색상 코드
ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[1;36m"
ANSI_YELLOW = "\033[1;33m"
ANSI_GREEN = "\033[1;32m"
ANSI_MAGENTA = "\033[1;35m"

# 각 깊이 수준에 대해 미리 정의된 색상 (ANSI 이스케이프 코드 사용)
DEPTH_COLORS = {
    1: "\033[96m",  # 밝은 청록색 (눈에 잘 띄는 첫 계층)
    2: "\033[93m",  # 노란색 (두 번째 계층)
    3: "\033[94m",  # 밝은 파란색 (세 번째 계층)
    4: "\033[95m",  # 보라색 (네 번째 계층)
    5: "\033[92m",  # 밝은 초록색 (다섯 번째 계층)
    "default": "\033[96m",
    "reset": "\033[0m",
}

# 스트림 모드 타입
StreamMode = Literal["messages", "updates", "values"]


# ============================================================
# 유틸리티 함수
# ============================================================
def random_uuid() -> str:
    """랜덤 UUID 문자열을 생성합니다."""
    return str(uuid.uuid4())


def get_role_from_messages(msg: BaseMessage) -> str:
    """메시지 객체에서 역할(role)을 추출합니다."""
    if isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, AIMessage):
        return "assistant"
    else:
        return "assistant"


def messages_to_history(messages: List[BaseMessage]) -> str:
    """메시지 리스트를 히스토리 문자열로 변환합니다."""
    return "\n".join(
        [f"{get_role_from_messages(msg)}: {msg.content}" for msg in messages]
    )


def format_namespace(namespace: tuple) -> str:
    """네임스페이스를 포맷팅합니다."""
    return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"


# ============================================================
# 메시지 텍스트 추출 유틸리티 (최신 LangChain v1.0+ 패턴 지원)
# ============================================================
@dataclass
class ParsedContentBlock:
    """파싱된 content_block 정보"""
    block_type: str  # "text", "reasoning", "tool_call", "tool_result", etc.
    content: Any
    metadata: Optional[Dict[str, Any]] = None


def parse_content_blocks(msg: Union[AIMessageChunk, BaseMessage, Any]) -> List[ParsedContentBlock]:
    """
    메시지의 content_blocks를 파싱합니다.
    
    LangChain v1.0+의 통합 content_blocks 형식을 지원합니다:
    - type: "text" - 텍스트 응답
    - type: "reasoning" - 모델의 추론 과정
    - type: "tool_call" - 도구 호출
    - type: "tool_result" - 도구 실행 결과
    
    Args:
        msg: 메시지 객체
        
    Returns:
        List[ParsedContentBlock]: 파싱된 블록 리스트
    """
    blocks = []
    
    # content_blocks 속성 확인 (최신 패턴)
    if hasattr(msg, "content_blocks") and msg.content_blocks:
        for block in msg.content_blocks:
            if isinstance(block, dict):
                block_type = block.get("type", "text")
                if block_type == "text":
                    blocks.append(ParsedContentBlock(
                        block_type="text",
                        content=block.get("text", "")
                    ))
                elif block_type == "reasoning":
                    blocks.append(ParsedContentBlock(
                        block_type="reasoning",
                        content=block.get("reasoning", "")
                    ))
                elif block_type == "tool_call":
                    blocks.append(ParsedContentBlock(
                        block_type="tool_call",
                        content=block.get("args", {}),
                        metadata={"name": block.get("name"), "id": block.get("id")}
                    ))
                else:
                    blocks.append(ParsedContentBlock(
                        block_type=block_type,
                        content=block
                    ))
            elif hasattr(block, "text"):
                blocks.append(ParsedContentBlock(
                    block_type="text",
                    content=block.text
                ))
    
    return blocks


def extract_message_text(msg: Union[AIMessageChunk, BaseMessage, Any]) -> str:
    """
    메시지에서 텍스트를 추출합니다.
    
    최신 LangChain v1.0+의 .text 속성과 content_blocks를 우선적으로 사용합니다.
    
    Args:
        msg: AIMessageChunk, BaseMessage, 또는 기타 메시지 객체
        
    Returns:
        str: 추출된 텍스트
    """
    # 최신 LangChain v1.0+: .text 속성 (권장)
    if hasattr(msg, "text") and msg.text:
        return msg.text
    
    # 최신 LangChain v1.0+: content_blocks 속성
    if hasattr(msg, "content_blocks") and msg.content_blocks:
        texts = []
        for block in msg.content_blocks:
            if isinstance(block, dict):
                block_type = block.get("type", "text")
                if block_type == "text":
                    texts.append(block.get("text", ""))
                # reasoning 타입은 별도 처리 (옵션)
            elif hasattr(block, "text"):
                texts.append(block.text)
        if texts:
            return "".join(texts)
    
    # 기존 방식: content 속성
    if hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)
            return "".join(texts)
    
    # 문자열인 경우
    if isinstance(msg, str):
        return msg
    
    return ""


def extract_reasoning(msg: Union[AIMessageChunk, BaseMessage, Any]) -> Optional[str]:
    """
    메시지에서 추론(reasoning) 내용을 추출합니다.
    
    LangChain v1.0+에서 일부 모델은 reasoning 블록을 반환합니다.
    
    Args:
        msg: 메시지 객체
        
    Returns:
        Optional[str]: 추론 내용 또는 None
    """
    if hasattr(msg, "content_blocks") and msg.content_blocks:
        for block in msg.content_blocks:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                return block.get("reasoning", "")
    return None


def extract_tool_calls(msg: Union[AIMessageChunk, BaseMessage, Any]) -> List[Dict[str, Any]]:
    """
    메시지에서 도구 호출 정보를 추출합니다.
    
    LangChain v1.0+의 tool_calls 및 content_blocks 패턴을 모두 지원합니다.
    
    Args:
        msg: 메시지 객체
        
    Returns:
        List[Dict[str, Any]]: 도구 호출 리스트
    """
    tool_calls = []
    
    # 방법 1: tool_calls 속성 (표준)
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_calls = list(msg.tool_calls)
    
    # 방법 2: content_blocks에서 tool_call 타입 추출
    if not tool_calls and hasattr(msg, "content_blocks") and msg.content_blocks:
        for block in msg.content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_call":
                tool_calls.append({
                    "name": block.get("name"),
                    "args": block.get("args", {}),
                    "id": block.get("id"),
                })
    
    return tool_calls


# ============================================================
# 스트리밍 응답 처리
# ============================================================
def stream_response(response, return_output: bool = False) -> Optional[str]:
    """
    AI 모델로부터의 응답을 스트리밍하여 각 청크를 처리하면서 출력합니다.
    
    최신 LangChain 패턴을 사용하여 청크를 누적합니다.

    Args:
        response (iterable): AIMessageChunk 객체 또는 문자열의 이터러블
        return_output (bool): True인 경우 연결된 응답 문자열 반환

    Returns:
        Optional[str]: return_output이 True인 경우 연결된 응답 문자열
    """
    # 최신 LangChain 패턴: 청크 누적
    full_message: Optional[AIMessageChunk] = None
    answer = ""
    
    for chunk in response:
        if isinstance(chunk, AIMessageChunk):
            # 최신 패턴: 청크 합산
            full_message = chunk if full_message is None else full_message + chunk
            text = extract_message_text(chunk)
            if text:
                answer += text
                print(text, end="", flush=True)
        elif isinstance(chunk, str):
            answer += chunk
            print(chunk, end="", flush=True)
    
    if return_output:
        return answer
    return None


async def astream_response(
    response: AsyncIterator, 
    return_output: bool = False
) -> Optional[str]:
    """
    AI 모델로부터의 응답을 비동기로 스트리밍합니다.

    Args:
        response: AsyncIterator of AIMessageChunk or strings
        return_output (bool): True인 경우 연결된 응답 문자열 반환

    Returns:
        Optional[str]: return_output이 True인 경우 연결된 응답 문자열
    """
    full_message: Optional[AIMessageChunk] = None
    answer = ""
    
    async for chunk in response:
        if isinstance(chunk, AIMessageChunk):
            full_message = chunk if full_message is None else full_message + chunk
            text = extract_message_text(chunk)
            if text:
                answer += text
                print(text, end="", flush=True)
        elif isinstance(chunk, str):
            answer += chunk
            print(chunk, end="", flush=True)
    
    if return_output:
        return answer
    return None


# ============================================================
# 콜백 함수 및 클래스
# ============================================================
def tool_callback(tool: Dict[str, Any]) -> None:
    """도구 호출 시 실행되는 콜백 함수입니다."""
    print(f"\n{ANSI_MAGENTA}[도구 호출]{ANSI_RESET}")
    tool_name = tool.get("tool") or tool.get("name", "unknown")
    print(f"Tool: {tool_name}")
    if tool_id := tool.get("id"):
        print(f"ID: {tool_id}")
    if tool_input := tool.get("tool_input") or tool.get("args"):
        if isinstance(tool_input, dict):
            for k, v in tool_input.items():
                print(f"  {k}: {v}")
        else:
            print(f"  Input: {tool_input}")


def observation_callback(observation: Dict[str, Any]) -> None:
    """관찰 결과(도구 실행 결과)를 출력하는 콜백 함수입니다."""
    print(f"\n{ANSI_YELLOW}[도구 결과]{ANSI_RESET}")
    if name := observation.get("name"):
        print(f"Tool: {name}")
    content = observation.get("observation", "")
    # 긴 결과는 잘라서 표시
    if len(str(content)) > 500:
        print(f"Result: {str(content)[:500]}...")
    else:
        print(f"Result: {content}")


def result_callback(result: str) -> None:
    """최종 결과를 출력하는 콜백 함수입니다."""
    print(f"\n{ANSI_GREEN}[최종 답변]{ANSI_RESET}")
    print(result)


@dataclass
class AgentCallbacks:
    """
    에이전트 콜백 함수들을 포함하는 데이터 클래스입니다.

    Attributes:
        tool_callback: 도구 사용 시 호출되는 콜백 함수
        observation_callback: 관찰 결과 처리 시 호출되는 콜백 함수
        result_callback: 최종 결과 처리 시 호출되는 콜백 함수
    """
    tool_callback: Callable[[Dict[str, Any]], None] = tool_callback
    observation_callback: Callable[[Dict[str, Any]], None] = observation_callback
    result_callback: Callable[[str], None] = result_callback


# ============================================================
# 에이전트 스트림 파서 (최신 LangChain v1.0+ 패턴)
# ============================================================
class AgentStreamParser:
    """
    에이전트의 스트림 출력을 파싱하고 처리하는 클래스입니다.
    
    LangChain v1.0+에서는 tool_calls 기반 패턴을 사용합니다.
    
    Example:
        ```python
        parser = AgentStreamParser()
        for chunk in agent.stream(inputs, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                parser.process_node_output(node_name, node_output)
        ```
    """

    def __init__(self, callbacks: Optional[AgentCallbacks] = None):
        """
        AgentStreamParser 객체를 초기화합니다.

        Args:
            callbacks: 파싱 과정에서 사용할 콜백 함수들
        """
        self.callbacks = callbacks or AgentCallbacks()
        self.output: Optional[str] = None

    def process_node_output(self, node_name: str, node_output: Any) -> None:
        """
        노드 출력을 처리합니다 (최신 LangGraph 패턴).
        
        Args:
            node_name: 노드 이름
            node_output: 노드 출력값
        """
        if isinstance(node_output, dict):
            messages = node_output.get("messages", [])
            if isinstance(messages, list):
                for msg in messages:
                    self._process_message(msg)
            elif messages:
                self._process_message(messages)

    def _process_message(self, msg: Any) -> None:
        """메시지를 처리합니다."""
        # AI 메시지의 tool_calls 처리
        if isinstance(msg, (AIMessage, AIMessageChunk)):
            tool_calls = extract_tool_calls(msg)
            if tool_calls:
                for tc in tool_calls:
                    self._process_tool_call(tc)
            else:
                # 최종 응답
                text = extract_message_text(msg)
                if text:
                    self._process_result(text)
        
        # Tool 메시지 (도구 실행 결과) 처리
        elif isinstance(msg, ToolMessage):
            self._process_observation(msg)

    def _process_tool_call(self, tool_call: Dict[str, Any]) -> None:
        """도구 호출을 처리합니다."""
        tool_action = {
            "tool": tool_call.get("name"),
            "tool_input": tool_call.get("args"),
            "id": tool_call.get("id"),
        }
        self.callbacks.tool_callback(tool_action)

    def _process_observation(self, tool_msg: ToolMessage) -> None:
        """도구 실행 결과(관찰)를 처리합니다."""
        observation_dict = {
            "observation": tool_msg.content,
            "tool_call_id": getattr(tool_msg, "tool_call_id", None),
            "name": getattr(tool_msg, "name", None),
        }
        self.callbacks.observation_callback(observation_dict)

    def _process_result(self, result: str) -> None:
        """최종 결과를 처리합니다."""
        self.callbacks.result_callback(result)
        self.output = result

    # 하위 호환성을 위한 레거시 메서드
    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        """
        에이전트의 단계를 처리합니다.
        
        Note: 이 메서드는 하위 호환성을 위해 유지됩니다.
              새로운 코드에서는 process_node_output을 사용하세요.
        """
        if "messages" in step:
            messages = step["messages"]
            if isinstance(messages, list):
                for msg in messages:
                    self._process_message(msg)
            else:
                self._process_message(messages)
        elif "output" in step:
            self._process_result(step["output"])


# ============================================================
# 메시지 출력 유틸리티
# ============================================================
def pretty_print_messages(messages: List[BaseMessage]) -> None:
    """메시지 리스트를 예쁘게 출력합니다."""
    for message in messages:
        message.pretty_print()


def is_terminal_dict(data: Any) -> bool:
    """말단 딕셔너리인지 확인합니다."""
    if not isinstance(data, dict):
        return False
    for value in data.values():
        if isinstance(value, (dict, list)) or hasattr(value, "__dict__"):
            return False
    return True


def format_terminal_dict(data: Dict[str, Any]) -> str:
    """말단 딕셔너리를 포맷팅합니다."""
    items = []
    for key, value in data.items():
        if isinstance(value, str):
            items.append(f'"{key}": "{value}"')
        else:
            items.append(f'"{key}": {value}')
    return "{" + ", ".join(items) + "}"


def _display_message_tree(
    data: Any, indent: int = 0, node: Optional[str] = None, is_root: bool = False
) -> None:
    """JSON 객체의 트리 구조를 타입 정보 없이 출력합니다."""
    spacing = " " * indent * 4
    color = DEPTH_COLORS.get(indent + 1, DEPTH_COLORS["default"])

    if isinstance(data, dict):
        if not is_root and node is not None:
            if is_terminal_dict(data):
                print(
                    f'{spacing}{color}{node}{DEPTH_COLORS["reset"]}: {format_terminal_dict(data)}'
                )
            else:
                print(f'{spacing}{color}{node}{DEPTH_COLORS["reset"]}:')
                for key, value in data.items():
                    _display_message_tree(value, indent + 1, key)
        else:
            for key, value in data.items():
                _display_message_tree(value, indent + 1, key)

    elif isinstance(data, list):
        if not is_root and node is not None:
            print(f'{spacing}{color}{node}{DEPTH_COLORS["reset"]}:')

        for index, item in enumerate(data):
            print(f'{spacing}    {color}index [{index}]{DEPTH_COLORS["reset"]}')
            _display_message_tree(item, indent + 1)

    elif hasattr(data, "__dict__") and not is_root:
        if node is not None:
            print(f'{spacing}{color}{node}{DEPTH_COLORS["reset"]}:')
        _display_message_tree(data.__dict__, indent)

    else:
        if node is not None:
            value_str = f'"{data}"' if isinstance(data, str) else str(data)
            print(f'{spacing}{color}{node}{DEPTH_COLORS["reset"]}: {value_str}')


def display_message_tree(message: Union[BaseMessage, Any]) -> None:
    """메시지 트리를 표시하는 주 함수입니다."""
    if isinstance(message, BaseMessage):
        _display_message_tree(message.__dict__, is_root=True)
    else:
        _display_message_tree(message, is_root=True)


# ============================================================
# Message Chunk Accumulator (최신 LangChain 패턴)
# ============================================================
class MessageChunkAccumulator:
    """
    메시지 청크를 누적하고 관리하는 클래스입니다.
    
    최신 LangChain 패턴에 따라 AIMessageChunk를 합산합니다.
    
    Example:
        ```python
        accumulator = MessageChunkAccumulator()
        for chunk in model.stream("Hello"):
            accumulator.add(chunk)
            print(accumulator.text)  # 누적된 텍스트
        full_message = accumulator.get_full_message()
        ```
    """

    def __init__(self):
        self._reset_state()

    def _reset_state(self) -> None:
        """상태 초기화"""
        self.gathered: Optional[AIMessageChunk] = None
        self.current_node: Optional[str] = None
        self.current_namespace: Optional[str] = None

    def add(
        self,
        chunk: AIMessageChunk,
        node: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """
        청크를 누적합니다.
        
        Args:
            chunk: 추가할 AI 메시지 청크
            node: 현재 노드명 (선택사항)
            namespace: 현재 네임스페이스 (선택사항)
        """
        # 노드/네임스페이스가 변경되면 리셋
        if self._should_reset(node, namespace):
            self._reset_state()

        self.current_node = node if node is not None else self.current_node
        self.current_namespace = namespace if namespace is not None else self.current_namespace

        # 최신 LangChain 패턴: 청크 합산
        self.gathered = chunk if self.gathered is None else self.gathered + chunk

    def _should_reset(self, node: Optional[str], namespace: Optional[str]) -> bool:
        """상태 리셋 여부 확인"""
        if node is None and namespace is None:
            return False
        if node is not None and self.current_node is not None and node != self.current_node:
            return True
        if namespace is not None and self.current_namespace is not None and namespace != self.current_namespace:
            return True
        return False

    @property
    def text(self) -> str:
        """현재까지 누적된 텍스트를 반환합니다."""
        if self.gathered is None:
            return ""
        return extract_message_text(self.gathered)

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """현재까지 누적된 도구 호출을 반환합니다."""
        if self.gathered is None:
            return []
        return extract_tool_calls(self.gathered)

    def get_full_message(self) -> Optional[AIMessageChunk]:
        """전체 누적된 메시지를 반환합니다."""
        return self.gathered

    def reset(self) -> None:
        """상태를 초기화합니다."""
        self._reset_state()


# 하위 호환성을 위한 별칭
ToolChunkHandler = MessageChunkAccumulator


# ============================================================
# 출력 헬퍼 함수
# ============================================================
def _print_node_header(
    node_name: str, namespace: Optional[tuple] = None, prev_node: str = ""
) -> None:
    """노드 헤더를 출력합니다."""
    if node_name == prev_node:
        return
    
    print("\n" + SEPARATOR_LINE)
    if namespace is None or format_namespace(namespace) == "root graph":
        print(f"🔄 Node: {ANSI_CYAN}{node_name}{ANSI_RESET} 🔄")
    else:
        formatted_namespace = format_namespace(namespace)
        print(
            f"🔄 Node: {ANSI_CYAN}{node_name}{ANSI_RESET} in [{ANSI_YELLOW}{formatted_namespace}{ANSI_RESET}] 🔄"
        )
    print(SEPARATOR_DASH)


def _print_chunk_content(chunk_msg: Any, show_reasoning: bool = False) -> None:
    """
    청크 메시지의 내용을 출력합니다.
    
    Args:
        chunk_msg: 출력할 청크 메시지
        show_reasoning: reasoning 블록 표시 여부 (기본: False)
    """
    # Reasoning 출력 (옵션)
    if show_reasoning:
        reasoning = extract_reasoning(chunk_msg)
        if reasoning:
            print(f"{ANSI_YELLOW}[Reasoning]{ANSI_RESET} {reasoning}", end="", flush=True)
    
    # 텍스트 출력
    text = extract_message_text(chunk_msg)
    if text:
        print(text, end="", flush=True)
    
    # 도구 호출 정보 출력
    tool_calls = extract_tool_calls(chunk_msg)
    if tool_calls:
        for tc in tool_calls:
            tool_name = tc.get("name", "unknown")
            print(f"\n{ANSI_MAGENTA}[Tool Call]{ANSI_RESET} {tool_name}", end="", flush=True)


def _print_base_message(msg: BaseMessage, streaming: bool = True) -> None:
    """BaseMessage를 출력합니다."""
    text = extract_message_text(msg)
    if text:
        if streaming:
            print(text, end="", flush=True)
        else:
            print(text)
    else:
        msg.pretty_print()


def _print_node_chunk(node_chunk: Any, streaming: bool = True) -> None:
    """
    노드 청크 데이터를 출력합니다.

    Args:
        node_chunk: 출력할 노드 청크
        streaming: 스트리밍 모드 여부 (True면 flush 사용)
    """
    if node_chunk is None:
        return

    if isinstance(node_chunk, dict):
        for k, v in node_chunk.items():
            if isinstance(v, BaseMessage):
                _print_base_message(v, streaming)
            elif isinstance(v, list):
                for list_item in v:
                    if isinstance(list_item, BaseMessage):
                        _print_base_message(list_item, streaming)
                    else:
                        text = extract_message_text(list_item)
                        if text:
                            print(text, end="" if streaming else "\n", flush=streaming)
                        else:
                            print(list_item, end="" if streaming else "\n", flush=streaming)
            elif isinstance(v, dict):
                for v_key, v_value in v.items():
                    print(f"{v_key}:\n{v_value}")
            else:
                if streaming:
                    print(v, end="", flush=True)
                else:
                    print(f"{ANSI_GREEN}{k}{ANSI_RESET}:\n{v}")
    elif hasattr(node_chunk, "__iter__") and not isinstance(node_chunk, str):
        try:
            for item in node_chunk:
                text = extract_message_text(item)
                if text:
                    print(text, end="" if streaming else "\n", flush=streaming)
                else:
                    print(item, end="" if streaming else "\n", flush=streaming)
        except TypeError:
            print(node_chunk, end="" if streaming else "\n", flush=streaming)
    else:
        print(node_chunk, end="" if streaming else "\n", flush=streaming)


# ============================================================
# 그래프 실행 함수 (동기)
# ============================================================
def stream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    node_names: Optional[List[str]] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stream_mode: StreamMode = "messages",
) -> Optional[Dict[str, Any]]:
    """
    LangGraph의 실행 결과를 스트리밍하여 출력하는 함수입니다.

    Args:
        graph: 실행할 컴파일된 LangGraph 객체
        inputs: 그래프에 전달할 입력값 딕셔너리
        config: 실행 설정
        node_names: 출력할 노드 이름 목록 (None이면 모든 노드 출력)
        callback: 각 청크 처리를 위한 콜백 함수
        stream_mode: 스트리밍 모드 ("messages", "updates", "values")

    Returns:
        Optional[Dict[str, Any]]: 최종 결과
    """
    config = config or {}
    node_names = node_names or []
    prev_node = ""
    final_result: Optional[Dict[str, Any]] = None

    if stream_mode == "messages":
        for chunk_msg, metadata in graph.stream(inputs, config, stream_mode="messages"):
            curr_node = metadata["langgraph_node"]
            final_result = {"node": curr_node, "content": chunk_msg, "metadata": metadata}

            if not node_names or curr_node in node_names:
                if callback:
                    callback({"node": curr_node, "content": chunk_msg})
                else:
                    _print_node_header(curr_node, prev_node=prev_node)
                    text = extract_message_text(chunk_msg)
                    if text:
                        print(text, end="", flush=True)
                prev_node = curr_node

    elif stream_mode == "values":
        # stream_mode="values": 각 단계의 전체 상태를 반환
        for chunk in graph.stream(inputs, config, stream_mode="values"):
            final_result = chunk
            if callback:
                callback({"content": chunk})
            else:
                # 최신 메시지 출력
                if "messages" in chunk and chunk["messages"]:
                    latest_msg = chunk["messages"][-1]
                    text = extract_message_text(latest_msg)
                    if text:
                        print(text, end="\n", flush=True)
                    # 도구 호출 표시
                    tool_calls = extract_tool_calls(latest_msg)
                    if tool_calls:
                        print(f"{ANSI_MAGENTA}Calling tools: {[tc.get('name') for tc in tool_calls]}{ANSI_RESET}")

    else:  # updates
        for namespace, chunk in graph.stream(
            inputs, config, stream_mode="updates", subgraphs=True
        ):
            for node_name, node_chunk in chunk.items():
                final_result = {"node": node_name, "content": node_chunk, "namespace": namespace}

                if node_names and node_name not in node_names:
                    continue

                if callback:
                    callback({"node": node_name, "content": node_chunk})
                else:
                    _print_node_header(node_name, namespace)
                    _print_node_chunk(node_chunk, streaming=False)
                    print(SEPARATOR_LINE)

    return final_result


def invoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    node_names: Optional[List[str]] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Optional[Dict[str, Any]]:
    """
    LangGraph 앱의 실행 결과를 스트리밍하여 출력하는 함수입니다.

    Args:
        graph: 실행할 컴파일된 LangGraph 객체
        inputs: 그래프에 전달할 입력값 딕셔너리
        config: 실행 설정
        node_names: 출력할 노드 이름 목록 (None이면 모든 노드 출력)
        callback: 각 청크 처리를 위한 콜백 함수

    Returns:
        Optional[Dict[str, Any]]: 최종 결과
    """
    return stream_graph(
        graph=graph,
        inputs=inputs,
        config=config,
        node_names=node_names,
        callback=callback,
        stream_mode="updates",
    )


# ============================================================
# 그래프 실행 함수 (비동기)
# ============================================================
async def astream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    node_names: Optional[List[str]] = None,
    callback: Optional[Callable] = None,
    stream_mode: StreamMode = "messages",
    include_subgraphs: bool = False,
    show_reasoning: bool = False,
) -> Dict[str, Any]:
    """
    LangGraph의 실행 결과를 비동기적으로 스트리밍하고 출력하는 함수입니다.
    
    LangChain v1.0+의 create_agent 및 최신 스트리밍 패턴을 지원합니다.

    Args:
        graph: 실행할 컴파일된 LangGraph 객체 또는 create_agent로 생성된 에이전트
        inputs: 그래프에 전달할 입력값 딕셔너리
        config: 실행 설정 (선택적)
        node_names: 출력할 노드 이름 목록 (None이면 모든 노드 출력)
        callback: 각 청크 처리를 위한 콜백 함수
        stream_mode: 스트리밍 모드 ("messages", "updates", "values")
        include_subgraphs: 서브그래프 포함 여부
        show_reasoning: reasoning 블록 표시 여부 (기본: False)

    Returns:
        Dict[str, Any]: 최종 결과

    Raises:
        ValueError: 유효하지 않은 stream_mode가 주어진 경우
    
    Example:
        ```python
        from langchain.agents import create_agent
        
        agent = create_agent(model="gpt-4o", tools=[...])
        result = await astream_graph(
            agent, 
            {"messages": [{"role": "user", "content": "Hello"}]},
            stream_mode="messages"
        )
        ```
    """
    config = config or {}
    node_names = node_names or []
    final_result: Dict[str, Any] = {}
    prev_node = ""

    if stream_mode == "messages":
        final_result = await _astream_messages_mode(
            graph, inputs, config, node_names, callback, prev_node, show_reasoning
        )
    elif stream_mode == "values":
        final_result = await _astream_values_mode(
            graph, inputs, config, node_names, callback
        )
    elif stream_mode == "updates":
        final_result = await _astream_updates_mode(
            graph, inputs, config, node_names, callback, include_subgraphs, prev_node
        )
    else:
        raise ValueError(
            f"Invalid stream_mode: {stream_mode}. Must be 'messages', 'updates', or 'values'."
        )

    return final_result


async def _astream_messages_mode(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str],
    callback: Optional[Callable],
    prev_node: str,
    show_reasoning: bool = False,
) -> Dict[str, Any]:
    """messages 모드로 스트리밍합니다."""
    final_result: Dict[str, Any] = {}

    async for chunk_msg, metadata in graph.astream(
        inputs, config, stream_mode="messages"
    ):
        curr_node = metadata["langgraph_node"]
        final_result = {"node": curr_node, "content": chunk_msg, "metadata": metadata}

        if not node_names or curr_node in node_names:
            if callback:
                result = callback({"node": curr_node, "content": chunk_msg})
                if hasattr(result, "__await__"):
                    await result
            else:
                _print_node_header(curr_node, prev_node=prev_node)
                _print_chunk_content(chunk_msg, show_reasoning=show_reasoning)

            prev_node = curr_node

    return final_result


async def _astream_values_mode(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str],
    callback: Optional[Callable],
) -> Dict[str, Any]:
    """values 모드로 스트리밍합니다."""
    final_result: Dict[str, Any] = {}

    async for chunk in graph.astream(inputs, config, stream_mode="values"):
        final_result = chunk

        if callback:
            result = callback({"content": chunk})
            if hasattr(result, "__await__"):
                await result
        else:
            # 최신 메시지 출력
            if "messages" in chunk and chunk["messages"]:
                latest_msg = chunk["messages"][-1]
                text = extract_message_text(latest_msg)
                if text:
                    print(text, end="\n", flush=True)
                tool_calls = extract_tool_calls(latest_msg)
                if tool_calls:
                    print(f"{ANSI_MAGENTA}Calling tools: {[tc.get('name') for tc in tool_calls]}{ANSI_RESET}")

    return final_result


async def _astream_updates_mode(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str],
    callback: Optional[Callable],
    include_subgraphs: bool,
    prev_node: str,
) -> Dict[str, Any]:
    """updates 모드로 스트리밍합니다."""
    final_result: Dict[str, Any] = {}

    async for chunk in graph.astream(
        inputs, config, stream_mode="updates", subgraphs=include_subgraphs
    ):
        # 반환 형식에 따라 처리 방법 분기
        if isinstance(chunk, tuple) and len(chunk) == 2:
            namespace, node_chunks = chunk
        else:
            namespace = ()
            node_chunks = chunk

        if isinstance(node_chunks, dict):
            for node_name, node_chunk in node_chunks.items():
                final_result = {
                    "node": node_name,
                    "content": node_chunk,
                    "namespace": namespace,
                }

                if node_names and node_name not in node_names:
                    continue

                if callback is not None:
                    result = callback({"node": node_name, "content": node_chunk})
                    if hasattr(result, "__await__"):
                        await result
                else:
                    _print_node_header(node_name, namespace, prev_node)
                    _print_node_chunk(node_chunk, streaming=True)

                prev_node = node_name
        else:
            print("\n" + SEPARATOR_LINE)
            print("🔄 Raw output 🔄")
            print(SEPARATOR_DASH)
            print(node_chunks, end="", flush=True)
            final_result = {"content": node_chunks}

    return final_result


async def ainvoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    node_names: Optional[List[str]] = None,
    callback: Optional[Callable] = None,
    include_subgraphs: bool = True,
) -> Dict[str, Any]:
    """
    LangGraph 앱의 실행 결과를 비동기적으로 스트리밍하여 출력하는 함수입니다.

    Args:
        graph: 실행할 컴파일된 LangGraph 객체
        inputs: 그래프에 전달할 입력값 딕셔너리
        config: 실행 설정 (선택적)
        node_names: 출력할 노드 이름 목록 (None이면 모든 노드 출력)
        callback: 각 청크 처리를 위한 콜백 함수
        include_subgraphs: 서브그래프 포함 여부

    Returns:
        Dict[str, Any]: 최종 결과
    """
    return await astream_graph(
        graph=graph,
        inputs=inputs,
        config=config,
        node_names=node_names,
        callback=callback,
        stream_mode="updates",
        include_subgraphs=include_subgraphs,
    )


# ============================================================
# 이벤트 스트리밍 (최신 LangChain astream_events 지원)
# ============================================================
async def astream_events(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    event_types: Optional[List[str]] = None,
    callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    LangGraph의 이벤트를 비동기적으로 스트리밍합니다.
    
    최신 LangChain의 astream_events API를 사용합니다.

    Args:
        graph: 실행할 컴파일된 LangGraph 객체
        inputs: 그래프에 전달할 입력값 딕셔너리
        config: 실행 설정 (선택적)
        event_types: 필터링할 이벤트 타입 목록
            - "on_chat_model_start": 모델 시작
            - "on_chat_model_stream": 토큰 스트리밍
            - "on_chat_model_end": 모델 완료
            - "on_tool_start": 도구 시작
            - "on_tool_end": 도구 완료
        callback: 각 이벤트 처리를 위한 콜백 함수

    Returns:
        Dict[str, Any]: 최종 결과
    """
    config = config or {}
    final_result: Dict[str, Any] = {}
    
    async for event in graph.astream_events(inputs, config, version="v2"):
        event_type = event.get("event", "")
        
        # 이벤트 타입 필터링
        if event_types and event_type not in event_types:
            continue
        
        final_result = event
        
        if callback:
            result = callback(event)
            if hasattr(result, "__await__"):
                await result
        else:
            # 기본 출력
            if event_type == "on_chat_model_start":
                print(f"{ANSI_CYAN}[Model Start]{ANSI_RESET} Input: {event.get('data', {}).get('input', '')[:50]}...")
            
            elif event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk:
                    text = extract_message_text(chunk)
                    if text:
                        print(text, end="", flush=True)
            
            elif event_type == "on_chat_model_end":
                output = event.get("data", {}).get("output")
                if output:
                    print(f"\n{ANSI_GREEN}[Model End]{ANSI_RESET}")
            
            elif event_type == "on_tool_start":
                tool_name = event.get("name", "unknown")
                print(f"\n{ANSI_MAGENTA}[Tool Start]{ANSI_RESET} {tool_name}")
            
            elif event_type == "on_tool_end":
                tool_name = event.get("name", "unknown")
                output = event.get("data", {}).get("output", "")
                print(f"{ANSI_MAGENTA}[Tool End]{ANSI_RESET} {tool_name}: {str(output)[:100]}...")

    return final_result
