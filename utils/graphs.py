"""
graphs.py - LangGraph 그래프 시각화 및 메시지 트리 출력 유틸리티

이 모듈은 LangGraph의 상태 그래프를 시각적으로 표현하고,
LangChain 메시지 객체를 계층적 트리 구조로 터미널에 출력하는 기능을 제공합니다.
"""

import logging
from typing import Any, Optional
from IPython.display import Image, display
from langgraph.graph.state import CompiledStateGraph
from dataclasses import dataclass
from langchain_core.messages import BaseMessage


# =============================================================================
# 그래프 시각화 관련 클래스 및 함수
# =============================================================================

@dataclass
class NodeStyles:
    """
    LangGraph 노드의 Mermaid 다이어그램 스타일을 정의하는 데이터클래스.
    
    Attributes:
        default: 일반 노드에 적용되는 기본 스타일 (청록색 배경, 굵은 글씨)
        first: 시작 노드에 적용되는 스타일 (연한 배경, 점선 테두리)
        last: 종료 노드에 적용되는 스타일 (진한 배경, 점선 테두리)
    """
    default: str = (
        "fill:#45C4B0, fill-opacity:0.3, color:#23260F, stroke:#45C4B0, "
        "stroke-width:1px, font-weight:bold, line-height:1.2"
    )
    first: str = (
        "fill:#45C4B0, fill-opacity:0.1, color:#23260F, stroke:#45C4B0, "
        "stroke-width:1px, font-weight:normal, font-style:italic, stroke-dasharray:2,2"
    )
    last: str = (
        "fill:#45C4B0, fill-opacity:1, color:#000000, stroke:#45C4B0, "
        "stroke-width:1px, font-weight:normal, font-style:italic, stroke-dasharray:2,2"
    )


def visualize_graph(graph: CompiledStateGraph, xray: bool = False) -> None:
    """
    LangGraph의 CompiledStateGraph를 Mermaid 다이어그램 이미지로 시각화합니다.
    
    Args:
        graph: 시각화할 LangGraph 컴파일된 상태 그래프 객체
        xray: True일 경우 서브그래프 내부까지 펼쳐서 표시 (기본값: False)
    
    Note:
        Jupyter Notebook 환경에서만 정상 동작합니다.
        graphviz 또는 mermaid 렌더링 의존성이 필요할 수 있습니다.
    """
    try:
        if isinstance(graph, CompiledStateGraph):
            display(
                Image(
                    graph.get_graph(xray=xray).draw_mermaid_png(
                        background_color="white",
                        node_colors=NodeStyles(),
                    )
                )
            )
    except Exception as e:
        logging.error(f"그래프 시각화 중 오류 발생: {e}")


# =============================================================================
# 메시지 트리 출력 관련 상수 및 함수
# =============================================================================

# 터미널 출력 시 깊이(depth)별 ANSI 색상 코드 매핑
# 트리의 계층 구조를 시각적으로 구분하기 위해 사용됨
DEPTH_COLORS = {
    1: "\033[96m",       # 밝은 청록색 (1단계 - 최상위)
    2: "\033[93m",       # 노란색 (2단계)
    3: "\033[94m",       # 파란색 (3단계)
    4: "\033[95m",       # 보라색 (4단계)
    5: "\033[92m",       # 밝은 초록색 (5단계)
    "default": "\033[96m",  # 5단계 초과 시 기본 색상
    "reset": "\033[0m",     # 색상 초기화 (기본 터미널 색상으로 복원)
}


def _is_terminal_dict(data: Any) -> bool:
    """
    딕셔너리가 '말단(terminal)' 딕셔너리인지 확인합니다.
    
    말단 딕셔너리란 모든 값이 원시 타입(primitive)이고,
    중첩된 dict, list, 또는 사용자 정의 객체를 포함하지 않는 딕셔너리입니다.
    
    Args:
        data: 확인할 데이터
        
    Returns:
        말단 딕셔너리이면 True, 아니면 False
    """
    if not isinstance(data, dict):
        return False
    for value in data.values():
        # 중첩 구조가 있으면 말단이 아님
        if isinstance(value, (dict, list)) or hasattr(value, "__dict__"):
            return False
    return True


def _format_terminal_dict(data: dict) -> str:
    """
    말단 딕셔너리를 한 줄의 문자열로 포맷팅합니다.
    
    Args:
        data: 포맷팅할 딕셔너리
        
    Returns:
        '{"key1": "value1", "key2": "value2"}' 형태의 문자열
    """
    items = []
    for key, value in data.items():
        # 모든 값을 문자열로 표시 (JSON 스타일)
        items.append(f'"{key}": "{value}"')
    return "{" + ", ".join(items) + "}"


def _display_message_tree(
    data: Any, 
    indent: int = 0, 
    node: Optional[str] = None, 
    is_root: bool = False
) -> None:
    """
    데이터를 재귀적으로 순회하며 계층적 트리 구조로 터미널에 출력합니다.
    
    각 깊이마다 다른 색상을 적용하여 구조를 시각적으로 파악하기 쉽게 합니다.
    dict, list, 사용자 정의 객체(__dict__ 포함), 원시 타입을 모두 처리합니다.
    
    Args:
        data: 출력할 데이터 (dict, list, object, 또는 원시 타입)
        indent: 현재 들여쓰기 깊이 (기본값: 0)
        node: 현재 노드의 키 이름 (기본값: None)
        is_root: 루트 노드 여부 - 루트는 키 이름 없이 출력 (기본값: False)
    """
    # 들여쓰기: 깊이당 4칸 공백
    spacing = " " * indent * 4
    # 현재 깊이에 해당하는 색상 선택
    color = DEPTH_COLORS.get(indent + 1, DEPTH_COLORS["default"])
    reset = DEPTH_COLORS["reset"]

    if isinstance(data, dict):
        # 딕셔너리 처리
        if not is_root and node is not None:
            if _is_terminal_dict(data):
                # 말단 딕셔너리는 한 줄로 출력
                print(f"{spacing}{color}{node}{reset}: {_format_terminal_dict(data)}")
            else:
                # 중첩 딕셔너리는 키를 출력하고 내부를 재귀 순회
                print(f"{spacing}{color}{node}{reset}:")
                for key, value in data.items():
                    _display_message_tree(value, indent + 1, key)
        else:
            # 루트 딕셔너리는 키 이름 없이 바로 내부 순회
            for key, value in data.items():
                _display_message_tree(value, indent + 1, key)

    elif isinstance(data, list):
        # 리스트 처리
        if not is_root and node is not None:
            print(f"{spacing}{color}{node}{reset}:")
        
        # 각 요소를 인덱스와 함께 출력
        for index, item in enumerate(data):
            print(f"{spacing}   {color}index [{index}]{reset}")
            _display_message_tree(item, indent + 1)
    
    elif hasattr(data, "__dict__") and not is_root:
        # 사용자 정의 객체 처리 (__dict__ 속성이 있는 경우)
        if node is not None:
            print(f"{spacing}{color}{node}{reset}:")
        # 객체의 __dict__를 재귀적으로 출력
        _display_message_tree(data.__dict__, indent)

    else:
        # 원시 타입 (str, int, float, bool, None 등) 처리
        if node is not None:
            if isinstance(data, str):
                value_str = f'"{data}"'  # 문자열은 따옴표로 감싸서 표시
            else:
                value_str = str(data)
            
            print(f"{spacing}{color}{node}{reset}: {value_str}")


def display_message_tree(message: Any) -> None:
    """
    LangChain 메시지 또는 임의의 데이터 구조를 트리 형태로 출력합니다.
    
    LangChain의 BaseMessage 객체(HumanMessage, AIMessage 등)를 받으면
    내부 속성들을 계층적으로 보기 좋게 터미널에 출력합니다.
    
    Args:
        message: 출력할 메시지 객체 또는 데이터 구조
        
    Example:
        >>> from langchain_core.messages import HumanMessage
        >>> msg = HumanMessage(content="안녕하세요")
        >>> display_message_tree(msg)
            content: "안녕하세요"
            type: "human"
            ...
    """
    if isinstance(message, BaseMessage):
        # BaseMessage 객체는 __dict__를 직접 순회
        _display_message_tree(message.__dict__, is_root=True)
    else:
        # 그 외 데이터는 그대로 전달
        _display_message_tree(message, is_root=True)