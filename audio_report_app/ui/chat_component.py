from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import streamlit as st
from interfaces.i_ui_component import IChatComponent


class ChatComponent(IChatComponent):
    """
    チャット機能用のUIコンポーネント

    チャットインターフェースの表示とユーザー入力の処理を提供します。
    """

    def __init__(self) -> None:
        """ChatComponentのインスタンスを初期化"""
        pass

    def render(self) -> Any:
        """
        UIコンポーネントをレンダリングする

        Returns:
            Any: レンダリング結果
        """
        st.subheader("💬 内容について質問する")
        return None

    def show_chat_interface(
        self, chat_history: List[Dict[str, str]], on_question_submit: Callable[[str], None]
    ) -> Optional[str]:
        """
        チャットインターフェースを表示する

        Args:
            chat_history: チャット履歴
            on_question_submit: 質問送信時のコールバック関数

        Returns:
            Optional[str]: 送信された質問（あれば）
        """
        # チャット履歴の表示
        for message in chat_history:
            if message["role"] == "user":
                st.markdown(f"🧑 **質問**: {message['content']}")
            else:
                st.markdown(f"🤖 **回答**: {message['content']}")

        # 質問入力フォーム
        user_question = st.text_input("質問を入力してください", key="user_question")

        # 送信ボタン
        if st.button("質問する"):
            if user_question:
                # コールバック関数を呼び出す
                on_question_submit(user_question)
                return user_question

        return None

    @classmethod
    def create(cls) -> IChatComponent:
        """
        ChatComponentのインスタンスを作成する

        Returns:
            IChatComponent: 新しいChatComponentインスタンス
        """
        return ChatComponent()
