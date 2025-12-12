"""
langsmith.py - LangSmith 트레이싱 설정 유틸리티

이 모듈은 LangChain 애플리케이션의 실행을 추적하고 디버깅하기 위한
LangSmith 서비스 연동을 설정하는 기능을 제공합니다.

LangSmith는 LangChain 팀에서 제공하는 관측성(observability) 플랫폼으로,
LLM 호출, 체인 실행, 에이전트 동작 등을 시각화하고 분석할 수 있습니다.
"""

import os
import logging
from typing import Optional


def langsmith(project_name: Optional[str] = None, set_enable: bool = True) -> None:
    """
    LangSmith 트레이싱을 활성화하거나 비활성화합니다.
    
    LangSmith를 활성화하면 LangChain 애플리케이션의 모든 실행 과정이
    LangSmith 대시보드에 자동으로 기록됩니다.
    
    Args:
        project_name: LangSmith 프로젝트 이름. 
                      동일한 프로젝트명으로 실행된 트레이스들이 그룹화됩니다.
                      set_enable=True일 경우 필수입니다.
        set_enable: True면 트레이싱 활성화, False면 비활성화 (기본값: True)
    
    Raises:
        ValueError: set_enable=True인데 LANGSMITH_API_KEY 환경변수가 설정되지 않은 경우
    
    Example:
        >>> # 트레이싱 활성화
        >>> langsmith(project_name="my-chatbot-project")
        
        >>> # 트레이싱 비활성화 (프로덕션 환경 등)
        >>> langsmith(set_enable=False)
    
    Note:
        사용 전 LANGSMITH_API_KEY 환경변수를 설정해야 합니다.
        API 키는 https://smith.langchain.com 에서 발급받을 수 있습니다.
    """
    if set_enable:
        # 환경변수에서 LangSmith API 키 가져오기
        langsmith_key = os.getenv("LANGSMITH_API_KEY", "")
        
        # API 키가 비어있으면 에러 발생
        if langsmith_key.strip() == "":
            raise ValueError(
                "LANGSMITH_API_KEY 환경변수가 설정되지 않았습니다. "
                "https://smith.langchain.com 에서 API 키를 발급받으세요."
            )

        # LangSmith 관련 환경변수 설정
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"  # API 엔드포인트
        os.environ["LANGSMITH_TRACING"] = "true"  # 트레이싱 활성화 플래그
        os.environ["LANGSMITH_PROJECT"] = project_name or "default"  # 프로젝트명 (없으면 기본값)
        
        logging.info(f"LangSmith 트레이싱이 활성화되었습니다. 프로젝트: {project_name}")

    else:
        # 트레이싱 비활성화 - LANGSMITH_TRACING을 false로 설정
        os.environ["LANGSMITH_TRACING"] = "false"
        logging.info("LangSmith 트레이싱이 비활성화되었습니다.")
