import gradio as gr
import requests
import json
import os
from typing import List, Tuple

class JapaneseLLMChat:
    def __init__(self):
        # 利用可能な日本語LLMモデル
        self.models = {
            "cyberagent/open-calm-7b": "CyberAgent Open CALM 7B",
            "rinna/japanese-gpt-neox-3.6b-instruction-sft": "Rinna GPT-NeoX 3.6B",
            "matsuo-lab/weblab-10b-instruction-sft": "Matsuo Lab WebLab 10B",
            "stabilityai/japanese-stablelm-instruct-alpha-7b": "Japanese StableLM 7B"
        }
        
        # デフォルトモデル
        self.current_model = "cyberagent/open-calm-7b"
        
        # HuggingFace API設定
        self.api_url = "https://api-inference.huggingface.co/models/"
        self.headers = {}
        
    def set_api_key(self, api_key: str):
        """APIキーを設定"""
        if api_key.strip():
            self.headers = {"Authorization": f"Bearer {api_key}"}
            return "✅ APIキーが設定されました"
        else:
            return "❌ 有効なAPIキーを入力してください"
    
    def set_model(self, model_name: str):
        """使用するモデルを変更"""
        self.current_model = model_name
        return f"モデルを {self.models[model_name]} に変更しました"
    
    def query_model(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """HuggingFace Inference APIにクエリを送信"""
        if not self.headers:
            return "❌ APIキーが設定されていません"
        
        url = self.api_url + self.current_model
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.95,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                    return generated_text.strip()
                else:
                    return "❌ 予期しないレスポンス形式です"
            elif response.status_code == 503:
                return "⏳ モデルが読み込み中です。しばらく待ってから再試行してください。"
            elif response.status_code == 401:
                return "❌ APIキーが無効です"
            else:
                return f"❌ エラーが発生しました (ステータス: {response.status_code})"
                
        except requests.exceptions.Timeout:
            return "⏳ リクエストがタイムアウトしました。再試行してください。"
        except requests.exceptions.RequestException as e:
            return f"❌ 接続エラー: {str(e)}"
    
    def chat_response(self, message: str, history: List[Tuple[str, str]], 
                     max_length: int, temperature: float) -> Tuple[str, List[Tuple[str, str]]]:
        """チャット応答を生成"""
        if not message.strip():
            return "", history
        
        # 対話履歴を考慮したプロンプト作成
        conversation_context = ""
        for user_msg, bot_msg in history[-3:]:  # 直近3回の会話を含める
            conversation_context += f"ユーザー: {user_msg}\nアシスタント: {bot_msg}\n"
        
        # プロンプトの構築
        if self.current_model == "rinna/japanese-gpt-neox-3.6b-instruction-sft":
            prompt = f"{conversation_context}ユーザー: {message}\nアシスタント:"
        elif "instruct" in self.current_model.lower():
            prompt = f"以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書いてください。\n\n### 指示:\n日本語で自然な会話を行ってください。\n\n### 入力:\n{conversation_context}ユーザー: {message}\n\n### 応答:\n"
        else:
            prompt = f"{conversation_context}ユーザー: {message}\nアシスタント:"
        
        # モデルから応答を取得
        response = self.query_model(prompt, max_length, temperature)
        
        # 履歴に追加
        history.append((message, response))
        
        return "", history

# チャットインスタンスを作成
chat_bot = JapaneseLLMChat()

# Gradio インターフェースの構築
def create_interface():
    with gr.Blocks(
        title="日本語LLMチャット",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1000px !important;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🤖 日本語LLMチャット
            HuggingFace Inference APIを使用した日本語対話システム
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # APIキー設定
                with gr.Group():
                    gr.Markdown("### 🔑 設定")
                    api_key_input = gr.Textbox(
                        label="HuggingFace API Token",
                        placeholder="hf_xxxxxxxxxxxxxxxxx",
                        type="password"
                    )
                    api_key_btn = gr.Button("APIキーを設定", variant="primary")
                    api_key_status = gr.Textbox(label="ステータス", interactive=False)
                
                # モデル選択
                with gr.Group():
                    gr.Markdown("### 🧠 モデル選択")
                    model_dropdown = gr.Dropdown(
                        choices=[(v, k) for k, v in chat_bot.models.items()],
                        value="cyberagent/open-calm-7b",
                        label="使用するモデル"
                    )
                    model_status = gr.Textbox(label="現在のモデル", interactive=False, 
                                            value=chat_bot.models[chat_bot.current_model])
                
                # パラメータ設定
                with gr.Group():
                    gr.Markdown("### ⚙️ 生成パラメータ")
                    max_length_slider = gr.Slider(
                        minimum=50, maximum=500, value=200,
                        label="最大生成長"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.7,
                        label="Temperature（創造性）"
                    )
            
            with gr.Column(scale=3):
                # チャットインターフェース
                chatbot = gr.Chatbot(
                    height=500,
                    label="会話",
                    show_label=True,
                    avatar_images=["👤", "🤖"]
                )
                
                msg = gr.Textbox(
                    label="メッセージ",
                    placeholder="メッセージを入力してください...",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("送信", variant="primary")
                    clear_btn = gr.Button("会話をクリア", variant="secondary")
        
        # 使用方法の説明
        with gr.Accordion("📖 使用方法", open=False):
            gr.Markdown(
                """
                1. **APIキーの設定**: HuggingFace（https://huggingface.co/settings/tokens）からAccess Tokenを取得し、上記フィールドに入力してください
                2. **モデル選択**: 使用したい日本語LLMを選択してください
                3. **パラメータ調整**: 必要に応じて生成パラメータを調整してください
                4. **チャット開始**: メッセージを入力して「送信」ボタンをクリックしてください
                
                **注意**: 
                - 初回使用時はモデルの読み込みに時間がかかる場合があります
                - 大きなモデル（7B以上）の使用には有料アカウントが必要な場合があります
                """
            )
        
        # イベントハンドラーの設定
        api_key_btn.click(
            chat_bot.set_api_key,
            inputs=[api_key_input],
            outputs=[api_key_status]
        )
        
        model_dropdown.change(
            chat_bot.set_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        send_btn.click(
            chat_bot.chat_response,
            inputs=[msg, chatbot, max_length_slider, temperature_slider],
            outputs=[msg, chatbot]
        )
        
        msg.submit(
            chat_bot.chat_response,
            inputs=[msg, chatbot, max_length_slider, temperature_slider],
            outputs=[msg, chatbot]
        )
        
        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, msg]
        )
    
    return demo

# アプリケーションの起動
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )