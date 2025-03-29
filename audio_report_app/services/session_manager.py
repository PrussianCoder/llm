from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
from interfaces.i_session_manager import ISessionManager
from pydub import AudioSegment
from utils.logging_config import LoggingConfig

# ロガーの取得
logger = LoggingConfig.get_logger("SessionManager")


class SessionManager(ISessionManager):
    """
    セッション状態を管理するクラス

    Streamlitのセッション状態を使用してアプリケーションの状態を管理します。
    """

    def __init__(self) -> None:
        """セッションマネージャーの初期化"""
        self.initialize_session()

    @classmethod
    def create(cls) -> ISessionManager:
        """
        SessionManagerのインスタンスを作成する

        Returns:
            ISessionManager: 新しいSessionManagerインスタンス
        """
        return cls()

    def initialize_session(self) -> None:
        """セッション状態を初期化する"""
        if "transcript" not in st.session_state:
            st.session_state.transcript = None
        if "chunks" not in st.session_state:
            st.session_state.chunks = None
        if "chunk_results" not in st.session_state:
            st.session_state.chunk_results = None
        if "last_file_name" not in st.session_state:
            st.session_state.last_file_name = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "settings" not in st.session_state:
            st.session_state.settings = {}

    def get_transcript(self) -> Optional[str]:
        """
        現在のセッションの文字起こし結果を取得する

        Returns:
            Optional[str]: 文字起こし結果（なければNone）
        """
        return st.session_state.transcript

    def set_transcript(self, transcript: str) -> None:
        """
        文字起こし結果をセッションに保存する

        Args:
            transcript: 文字起こし結果
        """
        st.session_state.transcript = transcript
        logger.info("文字起こし結果をセッションに保存しました")

    def get_chunks(self) -> Optional[List[AudioSegment]]:
        """
        現在のセッションの音声チャンクを取得する

        Returns:
            Optional[List[AudioSegment]]: 音声チャンク（なければNone）
        """
        return st.session_state.chunks

    def set_chunks(self, chunks: List[AudioSegment]) -> None:
        """
        音声チャンクをセッションに保存する

        Args:
            chunks: 音声チャンク
        """
        st.session_state.chunks = chunks
        logger.info(f"{len(chunks)}個の音声チャンクをセッションに保存しました")

    def get_chunk_results(self) -> Optional[List[str]]:
        """
        現在のセッションのチャンク毎の認識結果を取得する

        Returns:
            Optional[List[str]]: チャンク毎の認識結果（なければNone）
        """
        return st.session_state.chunk_results

    def set_chunk_results(self, chunk_results: List[str]) -> None:
        """
        チャンク毎の認識結果をセッションに保存する

        Args:
            chunk_results: チャンク毎の認識結果
        """
        st.session_state.chunk_results = chunk_results
        logger.info(f"{len(chunk_results)}個のチャンク認識結果をセッションに保存しました")

    def get_last_file_name(self) -> Optional[str]:
        """
        最後に処理したファイル名を取得する

        Returns:
            Optional[str]: 最後に処理したファイル名（なければNone）
        """
        return st.session_state.last_file_name

    def set_last_file_name(self, file_name: str) -> None:
        """
        最後に処理したファイル名をセッションに保存する

        Args:
            file_name: ファイル名
        """
        st.session_state.last_file_name = file_name
        logger.info(f"最後に処理したファイル名 '{file_name}' をセッションに保存しました")

    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        チャット履歴を取得する

        Returns:
            List[Dict[str, str]]: チャット履歴
        """
        return st.session_state.chat_history

    def add_chat_message(self, role: str, content: str) -> None:
        """
        チャットメッセージを追加する

        Args:
            role: メッセージの役割（"user" or "assistant"）
            content: メッセージの内容
        """
        st.session_state.chat_history.append({"role": role, "content": content})
        logger.info(f"チャットメッセージを追加しました: {role}")

    def is_new_file(self, file_name: str) -> bool:
        """
        新しいファイルかどうかを判定する

        Args:
            file_name: ファイル名

        Returns:
            bool: 新しいファイルかどうか
        """
        last_file = self.get_last_file_name()
        is_new = last_file is None or last_file != file_name
        if is_new:
            logger.info(f"新しいファイル '{file_name}' が検出されました")
        return is_new

    def get_settings(self) -> Dict[str, Any]:
        """
        現在のセッションの設定を取得する

        Returns:
            Dict[str, Any]: 設定値の辞書
        """
        return st.session_state.settings

    def set_settings(self, settings: Dict[str, Any]) -> None:
        """
        設定をセッションに保存する

        Args:
            settings: 設定値の辞書
        """
        st.session_state.settings = settings
        logger.info("設定をセッションに保存しました")
