from datetime import datetime 
import pytz 
import yfinance as yf 

def get_current_time(timezone: str = "Asia/Seoul"):
    tz = pytz.timezone(timezone)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    now_timezone = f"{now} {timezone}"
    return now_timezone

def get_stock_info(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info 
    return str(info)

def get_stock_history(ticker:str, period:str):
    stock = yf.Ticker(ticker)
    history = stock.history(period=period).to_markdown()
    return history

def get_stock_recommendation(ticker:str):
    stock = yf.Ticker(ticker)
    recommendation = stock.recommendations.to_markdown()
    return recommendation

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in the specified timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The timezone to get the current time in",
                    }
                },
                "required": ["timezone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_info",
            "description": "Get the information of the specified stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The ticker of the stock to get the information of",
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_history",
            "description": "Get the history of the specified stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The ticker of the stock to get the history of",
                    },
                    "period": {
                        "type": "string",
                        "description": "The period to get the history of (e.g. 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)",
                    }
                },
                "required": ["ticker", "period"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_recommendation",
            "description": "Get the recommendation of the specified stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The ticker of the stock to get the recommendation of",
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]

