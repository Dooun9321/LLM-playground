from tools import get_current_time, get_stock_info, get_stock_history, get_stock_recommendation, tools
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import streamlit as st
from collections import defaultdict

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_TEST_Key")

client = OpenAI(api_key=OPENAI_API_KEY)

def tool_list_to_tool_obj(tools):
    tool_calls_dict = defaultdict(
        lambda: {
            "id" : None, 
            "functions": {"arguments":"", "name":None}, 
            "type":None
        }
    )

    for tool_call in tools:
        # id가 None이 아닌 경우 설정
        if tool_call.id is not None:
            tool_calls_dict[tool_call.index]["id"] = tool_call.id

        # 함수 이름이 None이 아닌 경우 설정 
        if tool_call.function.name is not None:
            tool_calls_dict[tool_call.index]["functions"]["name"] = tool_call.function.name

        # 인자 추가
        tool_calls_dict[tool_call.index]["functions"]["arguments"] += tool_call.function.arguments

        # 함수 이름이 None이 아닌 경우 설정 
        if tool_call.type is not None:
            tool_calls_dict[tool_call.index]["type"] = tool_call.type

    return {"tool_calls": list(tool_calls_dict.values())}


def get_ai_response(messages, tools=None, stream=True):
    response = client.chat.completions.create(
        model="gpt-4o-mini",    # 응답생성에 사용할 모델 지정
        stream=stream,          # 스트리밍 모드 사용 여부
        messages=messages,      # 대화 기록을 입력으로 전달 
        tools=tools             # 사용할 수 있는 도구 목록 전달 
    )

    if stream:
        for chunk in response:
            yield chunk
    else:  
        return response

st.title("Chatbot with Tools")

# 초기 시스템 메세지
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"system", "content": "너는 사용자를 도와주는 상담사야. 사용자의 질문에 대해 적절한 답변을 해주세요."}
    ] 

# 대화 내용을 매번 출력
for msg in st.session_state.messages:
    if msg["role"] == "assistant" or msg["role"] == "user":
        st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():
    st.session_state.messages.append({
        "role":"user", "content":user_input
    })
    st.chat_message("user").write(user_input)

    ai_response = get_ai_response(st.session_state.messages, tools=tools, stream=True)

    content = ""
    tool_calls = []
    tool_calls_chunk = []  


    with st.chat_message("assistant").empty(): # 스트림릿 챗 메세지 초기화
        for chunk in ai_response:
            content_chunk = chunk.choices[0].delta.content
            if content_chunk:
                content += content_chunk
                st.markdown(content) # 스트림릿 챗 메시지에 마크다운으로 출력 

            if chunk.choices[0].delta.tool_calls:
                tool_calls_chunk += chunk.choices[0].delta.tool_calls

        tool_object = tool_list_to_tool_obj(tool_calls_chunk)
        tool_calls = tool_object["tool_calls"]

        if len(tool_calls) > 0:
            tool_call_msg = [tool_call["functions"] for tool_call in tool_calls]
            st.write(tool_call_msg)

    if tool_calls: # tool_calls가 있는 경우
        for tool_call in tool_calls:
            tool_name = tool_call["functions"]["name"]
            tool_call_id = tool_call["id"]
            arguments = json.loads(tool_call["functions"]["arguments"])


            if tool_name == "get_current_time":
                function_result = get_current_time(timezone=arguments['timezone'])

            elif tool_name == "get_stock_info":
                function_result = get_stock_info(ticker=arguments['ticker'])

            elif tool_name == "get_stock_history":
                function_result = get_stock_history(
                    ticker=arguments['ticker'], 
                    period=arguments['period']
                )

            elif tool_name == "get_stock_recommendation":
                function_result = get_stock_recommendation(ticker=arguments['ticker'])

            st.session_state.messages.append({
                "role":"function",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": function_result
            })

        st.session_state.messages.append({
            "role":"system",
            "content" : "이제 주어진 결과를 바탕으로 답변할 차례다"
        })
        ai_response = get_ai_response(st.session_state.messages, tools=tools, stream=True)

    content = ""
    with st.chat_message("assistant").empty():
        for chunk in ai_response:
            content_chunk = chunk.choices[0].delta.content
            if content_chunk:
                content += content_chunk
                st.markdown(content)

    st.session_state.messages.append({
        "role":"assistant", 
        "content": content
    })


