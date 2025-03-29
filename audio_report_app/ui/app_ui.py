from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
from pydub import AudioSegment

from ui.chat_component import ChatComponent
from ui.report_component import ReportComponent
from ui.sidebar_component import SidebarComponent
from ui.transcription_component import TranscriptionComponent


class AppUI:
    """
    アプリケーション全体のUIを管理するクラス

    各UIコンポーネントを組み合わせて、アプリケーション全体のUIを提供します。
    """

    def __init__(self) -> None:
        """AppUIのインスタンスを初期化"""
        self.sidebar_component = SidebarComponent.create()
        self.transcription_component = TranscriptionComponent.create()
        self.report_component = ReportComponent.create()
        self.chat_component = ChatComponent.create()

    def setup_sidebar(self) -> Dict[str, Any]:
        """
        サイドバーのUI要素をセットアップする

        Returns:
            Dict[str, Any]: 設定値の辞書
        """
        return self.sidebar_component.setup_sidebar()

    def get_api_key(self) -> str:
        """
        OpenAI APIキーを取得する

        Returns:
            str: APIキー
        """
        return self.sidebar_component.get_api_key()

    def file_upload(self, allowed_types: Optional[List[str]] = None) -> Any:
        """
        ファイルアップロードのUIを表示する

        Args:
            allowed_types: 許可するファイル拡張子のリスト

        Returns:
            Any: アップロードされたファイル
        """
        # デフォルトの許可タイプ
        if allowed_types is None:
            allowed_types = ["mp3", "mp4"]

        # 拡張子をフォーマット
        type_str = ", ".join([f".{t}" for t in allowed_types])
        description = f"音声/動画ファイルをアップロード ({type_str})"

        # ファイルアップローダーを表示
        uploaded_file = st.file_uploader(
            description,
            type=allowed_types,
            help="ファイルサイズによっては処理に時間がかかる場合があります",
        )

        return uploaded_file

    def show_transcription_result(self, transcript: str) -> None:
        """
        文字起こし結果を表示する

        Args:
            transcript: 文字起こし結果のテキスト
        """
        self.transcription_component.show_transcription_result(transcript)

    def show_chunk_details(self, chunks: List[AudioSegment], chunk_results: List[str]) -> None:
        """
        チャンク詳細を表示する

        Args:
            chunks: 音声チャンクのリスト
            chunk_results: チャンク毎の文字起こし結果のリスト
        """
        self.transcription_component.show_chunk_details(chunks, chunk_results)

    def update_realtime_results(
        self, chunk_text: str, chunk_index: int, total_chunks: int, chunk_duration: float
    ) -> None:
        """
        リアルタイム結果を更新する

        Args:
            chunk_text: チャンクの文字起こし結果
            chunk_index: チャンクのインデックス
            total_chunks: チャンクの総数
            chunk_duration: チャンクの長さ（秒）
        """
        self.transcription_component.update_realtime_results(
            chunk_text, chunk_index, total_chunks, chunk_duration
        )

    def show_realtime_transcription(
        self, chunks: List[AudioSegment], chunk_results: List[str]
    ) -> None:
        """
        リアルタイム処理後の最終結果を表示する

        Args:
            chunks: 音声チャンクのリスト
            chunk_results: チャンク毎の文字起こし結果のリスト
        """
        self.transcription_component.show_realtime_transcription(chunks, chunk_results)

    def show_report(self, report: str, report_type: str, file_name: str) -> None:
        """
        生成されたレポートを表示する

        Args:
            report: レポートのテキスト
            report_type: レポートタイプ
            file_name: 元のファイル名
        """
        self.report_component.show_report(report, report_type, file_name)

    def show_chat_interface(
        self, chat_history: List[Dict[str, str]], on_question_submit: callable
    ) -> Optional[str]:
        """
        チャットインターフェースを表示する

        Args:
            chat_history: チャット履歴
            on_question_submit: 質問送信時のコールバック関数

        Returns:
            Optional[str]: 送信された質問（あれば）
        """
        return self.chat_component.show_chat_interface(chat_history, on_question_submit)

    def show_error(self, error_message: str, include_traceback: bool = False) -> None:
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
                import traceback

                st.code(traceback.format_exc())
