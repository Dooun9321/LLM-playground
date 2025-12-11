from get_functions import get_current_time, tools
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_TEST_Key")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_ai_response(messages, tools=None):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )
    return response

messages = [
    {"role":"system", "content": "너는 사용자를 도와주는 상담사야. 사용자의 질문에 대해 적절한 답변을 해주세요."}
]

while True:
    user_input = input("사용자\t: ")

    if user_input == "exit":
        break

    messages.append({"role" : "user", "content" : user_input}) # 사용자 메시지를 대화 기록에 추가

    ai_response = get_ai_response(messages, tools=tools)
    ai_message = ai_response.choices[0].message
    print(ai_message)

    # AI 메시지를 먼저 추가 (tool_calls가 있든 없든)
    messages.append(ai_message)

    tool_calls = ai_message.tool_calls
    if tool_calls:
        tool_name = tool_calls[0].function.name
        tool_call_id = tool_calls[0].id

        if tool_name == "get_current_time":
            # tool 응답 메시지 추가
            messages.append({
                "role":"tool",
                "tool_call_id": tool_call_id,
                "content": get_current_time()
            }) 
        # 다시 AI에게 최종 응답 요청
        ai_response = get_ai_response(messages, tools=tools)
        ai_message = ai_response.choices[0].message
        messages.append(ai_message)
    
    print("AI\t: ", ai_message.content)