import os
from notion_client import Client
from openai import OpenAI

# 環境変数取得
notion = Client(auth=os.environ["NOTION_TOKEN"])
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DATABASE_ID = "あなたのNotion DB ID"

# ① Notionにアクセスできるか
db = notion.databases.retrieve(database_id=DATABASE_ID)
print("Notion OK:", db["title"][0]["plain_text"])

# ② OpenAIにアクセスできるか
resp = client.responses.create(
    model="gpt-4.1-mini",
    input="APIキーのテストです。OKとだけ返してください。"
)
print("OpenAI OK:", resp.output_text)
