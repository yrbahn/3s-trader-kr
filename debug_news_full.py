import requests
import json

def debug_news_api(code):
    url = f"https://m.stock.naver.com/api/news/stock/{code}?pageSize=3&page=1"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        print(json.dumps(data[0]['items'][0], indent=4, ensure_ascii=False))
    except Exception as e:
        print(e)

if __name__ == "__main__":
    debug_news_api("247540") # 에코프로비엠
