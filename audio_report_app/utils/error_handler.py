from __future__ import annotations

import traceback
from typing import Callable, Optional

import streamlit as st
from utils.logging_config import LoggingConfig

# ロガーの取得
logger = LoggingConfig.get_logger("ErrorHandler")


class ErrorHandler:
    """
    エラーハンドリングを行うユーティリティクラス

    統一されたエラー表示やエラーログ出力の機能を提供します。
    """

    @staticmethod
    def show_error(error_message: str, include_traceback: bool = False) -> None:
        """
        エラーメッセージを表示する

        Args:
            error_message: エラーメッセージ
            include_traceback: トレースバックを含めるかどうか
        """
        st.error(error_message)
        st.error(
            "処理中にエラーが発生しました。別のファイルを試すか、ファイルの品質を確認してください。"
        )

        # 詳細なエラー情報を開発者向けに表示（デバッグ用）
        if include_traceback:
            with st.expander("詳細なエラー情報（デバッグ用）"):
                st.code(traceback.format_exc())

    @staticmethod
    def with_error_handling(
        func: Callable, error_message: str = "エラーが発生しました", include_traceback: bool = True
    ) -> Callable:
        """
        関数をエラーハンドリングでラップする

        Args:
            func: ラップする関数
            error_message: エラー時に表示するメッセージ
            include_traceback: トレースバックを含めるかどうか

        Returns:
            Callable: エラーハンドリングでラップされた関数
        """

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ErrorHandler.show_error(f"{error_message}: {str(e)}", include_traceback)
                return None

        return wrapper

    def handle_error(
        self,
        error: Exception,
        error_message: Optional[str] = None,
        include_traceback: bool = False,
        show_ui_message: bool = True,
    ) -> None:
        """
        エラーを処理して適切にログとUIに表示する

        Args:
            error: 発生したエラー
            error_message: カスタムエラーメッセージ（指定しない場合はエラーのstr()を使用）
            include_traceback: トレースバックを含めるかどうか
            show_ui_message: UIにエラーメッセージを表示するかどうか
        """
        # エラーメッセージを取得
        message = error_message if error_message else str(error)

        # エラーをログに記録
        logger.error(message)

        # トレースバックをログに記録
        if include_traceback:
            tb = traceback.format_exc()
            logger.error(f"トレースバック:\n{tb}")

        # UIにエラーメッセージを表示
        if show_ui_message:
            self.show_error_message(message, include_traceback=include_traceback)

    def show_error_message(self, message: str, include_traceback: bool = False) -> None:
        """
        エラーメッセージをUIに表示する

        Args:
            message: 表示するエラーメッセージ
            include_traceback: トレースバックも表示するかどうか
        """
        st.error(f"エラーが発生しました: {message}")

        if include_traceback:
            with st.expander("詳細情報"):
                st.code(traceback.format_exc())

    def get_friendly_error_message(self, error: Exception) -> str:
        """
        ユーザーフレンドリーなエラーメッセージを取得する

        一般的なエラーに対して、技術的ではなくユーザーが理解しやすいメッセージを提供します。

        Args:
            error: エラーオブジェクト

        Returns:
            str: ユーザーフレンドリーなエラーメッセージ
        """
        error_type = type(error).__name__

        # エラータイプに基づいてフレンドリーなメッセージを返す
        friendly_messages = {
            "FileNotFoundError": "ファイルが見つかりませんでした。ファイルパスを確認してください。",
            "PermissionError": "ファイルへのアクセス権限がありません。",
            "TimeoutError": (
                "処理がタイムアウトしました。ネットワーク接続を確認するか、後でもう一度お試しください。"
            ),
            "ConnectionError": (
                "ネットワーク接続に問題があります。インターネット接続を確認してください。"
            ),
            "ValueError": "不正な値が入力されました。入力内容を確認してください。",
            "ImportError": (
                "必要なライブラリがインストールされていません。管理者に連絡してください。"
            ),
            "KeyError": "必要なキーが見つかりません。設定を確認してください。",
            "AttributeError": "必要な属性にアクセスできません。設定を確認してください。",
        }

        # 対応するフレンドリーメッセージがあれば返す、なければデフォルトメッセージを返す
        return friendly_messages.get(error_type, f"エラーが発生しました: {str(error)}")
