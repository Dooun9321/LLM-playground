from datetime import datetime

def get_current_time():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return now

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time",
        }
    }
]
    

if __name__ == "__main__":
    get_current_time()