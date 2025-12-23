import os
import json
import time
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typing import List, Dict, Optional
import requests
from notion_client import Client
import openai
from github import Github
import numpy as np

class ZettelkastenAutomation:
    """Zettelkastenメモの自動化システム（改良版）"""
    
    def __init__(self, notion_token: str, openai_api_key: str, github_token: str, 
                 database_id: str, repo_name: str, log_file: str = "processing_log.json"):
        """
        初期化
        
        Args:
            notion_token: Notion APIトークン
            openai_api_key: OpenAI APIキー
            github_token: GitHub Personal Access Token
            database_id: NotionデータベースID
            repo_name: GitHubリポジトリ名 (例: "username/repo")
            log_file: 処理ログファイルのパス
        """
        self.notion = Client(auth=notion_token)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.github = Github(github_token)
        self.database_id = database_id
        self.repo = self.github.get_repo(repo_name)
        
        # --- 修正箇所：logsフォルダへのパス設定 ---
        log_dir = "logs"
        # logsフォルダがなければ作成する
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"📁 フォルダを作成しました: {log_dir}")
            
        # フォルダ名とファイル名を結合する
        self.log_file = os.path.join(log_dir, log_file)
        # ---------------------------------------
        
        # 全ページのキャッシュ（関連メモ検索用）
        self.all_pages_cache = []
        
        # 処理ログを読み込み
        self.processing_log = self._load_log()
        
    def _load_log(self) -> Dict:
        """処理ログを読み込み"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"processed_pages": {}}
    
    def _save_log(self):
        """処理ログを保存"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.processing_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  ログ保存エラー: {e}")
    
    def _is_already_processed(self, page_id: str, last_edited_time: str) -> bool:
        """ページが既に処理済みかチェック"""
        if page_id in self.processing_log["processed_pages"]:
            log_entry = self.processing_log["processed_pages"][page_id]
            # 最終編集時刻が変わっていなければ処理済み
            return log_entry.get("last_edited_time") == last_edited_time
        return False
    
    def _add_to_log(self, page_id: str, title: str, last_edited_time: str, status: str = "success"):
        """処理ログに追加"""
        self.processing_log["processed_pages"][page_id] = {
            "title": title,
            "processed_at": datetime.now().isoformat(),
            "last_edited_time": last_edited_time,
            "status": status
        }
        self._save_log()
    
    def get_unprocessed_pages(self) -> List[Dict]:
        """未処理のページを取得"""
        print("📚 Notionデータベースから未処理ページを取得中...")
        pages = []
        has_more = True
        start_cursor = None
        
        while has_more:
            # Statusが「未処理」またはAI処理済みがFalseのページを取得
            response = self.notion.databases.query(
                database_id=self.database_id,
                start_cursor=start_cursor,
                filter={
                    "or": [
                        {
                            "property": "Status",
                            "select": {
                                "equals": "未処理"
                            }
                        },
                        {
                            "property": "AI処理済み",
                            "checkbox": {
                                "equals": False
                            }
                        }
                    ]
                }
            )
            pages.extend(response['results'])
            has_more = response['has_more']
            start_cursor = response.get('next_cursor')
            time.sleep(0.3)  # API制限対策
        
        # 最終編集時刻をチェックして未処理のものだけフィルタ
        unprocessed = []
        for page in pages:
            last_edited = page['last_edited_time']
            if not self._is_already_processed(page['id'], last_edited):
                unprocessed.append(page)
        
        print(f"✅ {len(unprocessed)}ページが未処理です（全{len(pages)}ページ中）")
        return unprocessed
    
    def get_all_pages(self) -> List[Dict]:
        """データベースから全ページを取得（関連メモ検索用）"""
        print("📚 全ページをキャッシュ中...")
        pages = []
        has_more = True
        start_cursor = None
        
        while has_more:
            response = self.notion.databases.query(
                database_id=self.database_id,
                start_cursor=start_cursor
            )
            pages.extend(response['results'])
            has_more = response['has_more']
            start_cursor = response.get('next_cursor')
            time.sleep(0.3)
            
        self.all_pages_cache = pages
        print(f"✅ {len(pages)}ページをキャッシュしました")
        return pages
    
    def get_page_content(self, page_id: str) -> str:
        """ページの本文を取得"""
        blocks = self.notion.blocks.children.list(block_id=page_id)
        content = []
        
        for block in blocks['results']:
            block_type = block['type']
            if block_type == 'paragraph':
                text = self._extract_text(block['paragraph'])
                if text:
                    content.append(text)
            elif block_type in ['heading_1', 'heading_2', 'heading_3']:
                text = self._extract_text(block[block_type])
                if text:
                    content.append(f"\n## {text}\n")
            elif block_type == 'bulleted_list_item':
                text = self._extract_text(block['bulleted_list_item'])
                if text:
                    content.append(f"- {text}")
            elif block_type == 'numbered_list_item':
                text = self._extract_text(block['numbered_list_item'])
                if text:
                    content.append(f"1. {text}")
            elif block_type == 'to_do':
                text = self._extract_text(block['to_do'])
                checked = "✓" if block['to_do'].get('checked') else "☐"
                if text:
                    content.append(f"{checked} {text}")
                    
        return '\n'.join(content)
    
    def _extract_text(self, block_content: Dict) -> str:
        """ブロックからテキストを抽出"""
        if 'rich_text' not in block_content:
            return ''
        return ''.join([text['plain_text'] for text in block_content['rich_text']])
    
    def analyze_with_ai(self, content: str, existing_tags: List[str] = None) -> Dict:
        """AIでメモを解析してタイトル・タグ・要約を生成"""
        print("🤖 AIで解析中...")
        
        existing_tags_str = ', '.join(existing_tags) if existing_tags else 'なし'
        
        prompt = f"""以下のZettelkastenメモを解析してください。

【メモ内容】
{content[:3000]}  # 長すぎる場合は切り詰め

【既存のタグ】
{existing_tags_str}

以下の形式でJSONを返してください：
{{
  "title": "「〜〜は〇〇である」や「～なのはなぜか」といった形式の簡潔なタイトル（50文字以内）",
  "tags": ["タグ1", "タグ2", "タグ3"],  // 3-5個のタグ
  "summary": "100文字程度の要約",
  "keywords": ["キーワード1", "キーワード2", "キーワード3"]  // 関連メモ検索用の重要キーワード
}}

【ルール】
- タイトルは必ず「〜は〇〇である」「〜について」「〜なのはなぜか」などの形式で、内容の本質を表現
- タグは既存タグも考慮しつつ、内容に最も適切なものを選択
- keywordsは、他のメモとリンクしやすくするために、文章ではなく具体的で短い名詞（固有名詞、技術用語、概念）を5〜8個程度抽出してください。出"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "あなたはZettelkasten方式の知識管理の専門家です。メモの本質を捉え、適切なタイトルとタグを付けることが得意です。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            print(f"✅ タイトル: {result['title']}")
            print(f"✅ タグ: {', '.join(result['tags'])}")
            print(f"✅ キーワード: {', '.join(result.get('keywords', []))}")
            return result
            
        except Exception as e:
            print(f"❌ AI解析エラー: {e}")
            return {
                "title": "無題のメモ",
                "tags": existing_tags or ["未分類"],
                "summary": content[:100],
                "keywords": []
            }
    
    def find_related_pages(self, keywords: List[str], current_page_id: str, 
                          top_k: int = 5) -> List[Dict]:
        """関連するページを検索"""
        print("🔍 関連メモを検索中...")
        
        if not keywords:
            return []
        
        related = []
        
        for page in self.all_pages_cache:
            if page['id'] == current_page_id:
                continue
            
            # タイトルを取得
            title = self._get_page_title(page)
            if not title or title == '無題':
                continue
            
            # タグを取得
            page_tags = self._get_page_tags(page)
            
            # スコア計算
            title_lower = title.lower()
            tags_lower = ' '.join(page_tags).lower()
            
            # タイトルでのキーワードマッチ
            title_matches = sum(1 for kw in keywords if kw.lower() in title_lower)
            
            # タグでのキーワードマッチ
            tag_matches = sum(1 for kw in keywords if kw.lower() in tags_lower)
            
            # 総合スコア（タイトルマッチを重視）
            score = title_matches * 2 + tag_matches
            
            if score > 0:
                related.append({
                    'id': page['id'],
                    'title': title,
                    'score': score
                })
        
        # スコア順にソート
        related.sort(key=lambda x: x['score'], reverse=True)
        
        if related:
            print(f"✅ {len(related[:top_k])}件の関連メモを発見")
            for i, rp in enumerate(related[:top_k], 1):
                print(f"   {i}. {rp['title']} (スコア: {rp['score']})")
        else:
            print("ℹ️  関連メモが見つかりませんでした")
        
        return related[:top_k]
    
    def _get_page_title(self, page: Dict) -> str:
        """ページのタイトルを取得"""
        try:
            title_property = page['properties'].get('Name') or page['properties'].get('title')
            if title_property and title_property['type'] == 'title':
                return ''.join([t['plain_text'] for t in title_property['title']])
        except:
            pass
        return ''
    
    def _get_page_tags(self, page: Dict) -> List[str]:
        """ページのタグを取得"""
        try:
            tags_prop = page['properties'].get('Tags', {})
            if tags_prop.get('multi_select'):
                return [tag['name'] for tag in tags_prop['multi_select']]
        except:
            pass
        return []
    
    def update_notion_page(self, page_id: str, title: str, tags: List[str], 
                          related_pages: List[Dict]):
        """Notionページを更新"""
        print("📝 Notionページを更新中...")
        
        try:
            # タイトル、タグ、Status、AI処理済みフラグを更新
            properties = {
                'Name': {'title': [{'text': {'content': title}}]},
                'Tags': {'multi_select': [{'name': tag} for tag in tags]},
                'Status': {'select': {'name': '処理済み'}},
                'AI処理済み': {'checkbox': True}
            }
            
            self.notion.pages.update(
                page_id=page_id,
                properties=properties
            )
            
            # 関連リンクを追加
            if related_pages:
                # 既存のブロックを取得
                existing_blocks = self.notion.blocks.children.list(block_id=page_id)
                
                # 「関連メモ」セクションが既に存在するかチェック
                has_related_section = False
                for block in existing_blocks['results']:
                    if block['type'] in ['heading_2', 'heading_3']:
                        text = self._extract_text(block[block['type']])
                        if '関連メモ' in text:
                            has_related_section = True
                            break
                
                if not has_related_section:
                    children = [
                        {
                            'object': 'block',
                            'type': 'divider',
                            'divider': {}
                        },
                        {
                            'object': 'block',
                            'type': 'heading_2',
                            'heading_2': {
                                'rich_text': [{'type': 'text', 'text': {'content': '🔗 関連メモ'}}]
                            }
                        }
                    ]
                    
                    for rp in related_pages:
                        children.append({
                            'object': 'block',
                            'type': 'paragraph',
                            'paragraph': {
                                'rich_text': [
                                    {'type': 'text', 'text': {'content': '→ '}},
                                    {'type': 'mention', 'mention': {'type': 'page', 'page': {'id': rp['id']}}}
                                ]
                            }
                        })
                    
                    self.notion.blocks.children.append(block_id=page_id, children=children)
            
            print("✅ Notionページの更新完了")
            
        except Exception as e:
            print(f"❌ Notion更新エラー: {e}")
            raise
    
    def convert_to_markdown(self, page: Dict, content: str, tags: List[str], 
                           related_pages: List[Dict]) -> str:
        """Markdown形式に変換（Obsidianプロパティ対応版）"""
        title = self._get_page_title(page)
        created_time = page['created_time'][:10]
        
        # --- 修正箇所：タグの処理 ---
        # 1. 各タグから「#」を除去
        # 2. YAMLのリスト形式（一行ずつ）にする
        yaml_tags = ""
        if tags:
            yaml_tags = "\ntags:\n" + "\n".join([f"  - {tag.replace('#', '')}" for tag in tags])
        
        # YAMLフロントマウントの組み立て
        # titleやdateの後のスペースも確実に確保
        md = f"""---
title: {title}
date: {created_time}{yaml_tags}
---

# {title}

{content}
"""
        
        if related_pages:
            md += "\n---\n\n## 🔗 関連メモ\n\n"
            for rp in related_pages:
                md += f"- [[{rp['title']}]]\n"
        
        return md
    
    def save_to_github(self, filename: str, content: str, commit_message: str):
        """GitHubにMarkdownファイルを保存"""
        print(f"📤 GitHubに保存中: {filename}")
        
        try:
            # ファイルが存在するか確認
            try:
                file = self.repo.get_contents(filename)
                # 既存ファイルを更新
                self.repo.update_file(
                    path=filename,
                    message=commit_message,
                    content=content,
                    sha=file.sha
                )
                print("✅ ファイルを更新しました")
            except:
                # 新規ファイルを作成
                self.repo.create_file(
                    path=filename,
                    message=commit_message,
                    content=content
                )
                print("✅ 新規ファイルを作成しました")
                
        except Exception as e:
            print(f"❌ GitHub保存エラー: {e}")
            raise
    
    def process_page(self, page_id: str):
        """1ページを処理"""
        print(f"\n{'='*60}")
        print(f"処理開始: {page_id}")
        print('='*60)
        
        # ページ情報を取得
        page = self.notion.pages.retrieve(page_id=page_id)
        existing_title = self._get_page_title(page)
        last_edited_time = page['last_edited_time']
        
        # 本文を取得
        content = self.get_page_content(page_id)
        if not content or len(content) < 20:
            print("⏭️  スキップ: 本文が短すぎます（20文字未満）")
            return
        
        # 既存のタグを取得
        existing_tags = self._get_page_tags(page)
        
        # AIで解析
        analysis = self.analyze_with_ai(content, existing_tags)
        
        # 関連ページを検索
        related_pages = self.find_related_pages(
            analysis.get('keywords', []), 
            page_id
        )
        
        # Notionを更新
        self.update_notion_page(
            page_id,
            analysis['title'],
            analysis['tags'],
            related_pages
        )
        
        # Markdownに変換
        markdown = self.convert_to_markdown(
            page,
            content,
            analysis['tags'],
            related_pages
        )

        # --- 修正箇所：保存先フォルダ名の変更 ---
        target_dir = "zettelkasten-vault" # フォルダ名をここで指定

        # GitHubに保存
        safe_title = analysis['title'].replace('/', '-').replace('\\', '-')[:50]
        # ファイル名に使えない文字を削除
        safe_title = ''.join(c for c in safe_title if c.isalnum() or c in (' ', '-', '_'))
        
        # フォルダ名を zettelkasten から zettelkasten-vault に変更
        filename = f"{target_dir}/{page['created_time'][:10]}_{safe_title}.md"
        
        self.save_to_github(
            filename,
            markdown,
            f"✨ Add: {analysis['title']}"
        )
        
        # ログに記録
        self._add_to_log(page_id, analysis['title'], last_edited_time, "success")
        
        print(f"✅ 処理完了: {analysis['title']}")
        time.sleep(1)  # API制限対策
    
    def run(self, limit: Optional[int] = None, force_reprocess: bool = False):
        """全ページを処理
        
        Args:
            limit: 処理するページ数の上限（Noneの場合は全て）
            force_reprocess: Trueの場合、処理済みページも再処理
        """
        print("\n🚀 Zettelkasten自動化システム開始\n")
        print(f"📅 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📝 ログファイル: {self.log_file}\n")
        
        # 全ページを取得（関連メモ検索用）
        self.get_all_pages()
        
        # 未処理ページを取得
        if force_reprocess:
            print("⚠️  強制再処理モード: 全ページを処理対象とします")
            pages = self.all_pages_cache
        else:
            pages = self.get_unprocessed_pages()
        
        if not pages:
            print("✨ 処理対象のページがありません")
            return
        
        # 処理対象を制限
        if limit:
            pages = pages[:limit]
            print(f"ℹ️  処理を{limit}ページに制限します")
        
        print(f"\n📊 処理対象: {len(pages)}ページ\n")
        
        # 各ページを処理
        success_count = 0
        error_count = 0
        
        for i, page in enumerate(pages, 1):
            print(f"\n進捗: {i}/{len(pages)}")
            try:
                self.process_page(page['id'])
                success_count += 1
            except Exception as e:
                print(f"❌ エラー: {e}")
                error_count += 1
                # エラーもログに記録
                self._add_to_log(
                    page['id'], 
                    self._get_page_title(page) or "タイトル取得失敗",
                    page['last_edited_time'],
                    f"error: {str(e)}"
                )
                continue
        
        # 処理結果のサマリー
        print("\n" + "="*60)
        print("🎉 処理完了")
        print("="*60)
        print(f"✅ 成功: {success_count}ページ")
        print(f"❌ エラー: {error_count}ページ")
        print(f"📊 合計: {len(pages)}ページ")
        print("="*60)


def main():
    """メイン実行関数"""
    
    # 環境変数から認証情報を取得
    NOTION_TOKEN = os.getenv('NOTION_TOKEN')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
    DATABASE_ID = os.getenv('NOTION_DATABASE_ID')
    REPO_NAME = os.getenv('GITHUB_REPO')  # 例: "username/zettelkasten"
    
    if not all([NOTION_TOKEN, OPENAI_API_KEY, GITHUB_TOKEN, DATABASE_ID, REPO_NAME]):
        print("❌ 環境変数が設定されていません")
        print("\n必要な環境変数:")
        print("  - NOTION_TOKEN")
        print("  - OPENAI_API_KEY")
        print("  - GITHUB_TOKEN")
        print("  - NOTION_DATABASE_ID")
        print("  - GITHUB_REPO")
        return
    
    # システムを初期化
    system = ZettelkastenAutomation(
        notion_token=NOTION_TOKEN,
        openai_api_key=OPENAI_API_KEY,
        github_token=GITHUB_TOKEN,
        database_id=DATABASE_ID,
        repo_name=REPO_NAME,
        log_file="zettelkasten_processing_log.json"
    )
    
    # 実行
    # 初回テスト時はlimit=5などで制限推奨
    # force_reprocess=Trueで強制再処理
    system.run(limit=None, force_reprocess=False)


if __name__ == '__main__':
    main()