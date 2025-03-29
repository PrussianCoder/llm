from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from pydub import AudioSegment


class ISessionManager(ABC):
    """
    セッション状態を管理するインターフェース

    アプリケーションの状態（文字起こし結果、チャットの履歴など）を管理します。
    """

    @abstractmethod
    def initialize_session(self) -> None:
        """
        セッション状態を初期化する
        """
        pass

    @abstractmethod
    def get_transcript(self) -> Optional[str]:
        """
        現在のセッションの文字起こし結果を取得する

        Returns:
            Optional[str]: 文字起こし結果（なければNone）
        """
        pass

    @abstractmethod
    def set_transcript(self, transcript: str) -> None:
        """
        文字起こし結果をセッションに保存する

        Args:
            transcript: 文字起こし結果
        """
        pass

    @abstractmethod
    def get_chunks(self) -> Optional[List[AudioSegment]]:
        """
        現在のセッションの音声チャンクを取得する

        Returns:
            Optional[List[AudioSegment]]: 音声チャンク（なければNone）
        """
        pass

    @abstractmethod
    def set_chunks(self, chunks: List[AudioSegment]) -> None:
        """
        音声チャンクをセッションに保存する

        Args:
            chunks: 音声チャンク
        """
        pass

    @abstractmethod
    def get_chunk_results(self) -> Optional[List[str]]:
        """
        現在のセッションのチャンク毎の認識結果を取得する

        Returns:
            Optional[List[str]]: チャンク毎の認識結果（なければNone）
        """
        pass

    @abstractmethod
    def set_chunk_results(self, chunk_results: List[str]) -> None:
        """
        チャンク毎の認識結果をセッションに保存する

        Args:
            chunk_results: チャンク毎の認識結果
        """
        pass

    @abstractmethod
    def get_last_file_name(self) -> Optional[str]:
        """
        最後に処理したファイル名を取得する

        Returns:
            Optional[str]: 最後に処理したファイル名（なければNone）
        """
        pass

    @abstractmethod
    def set_last_file_name(self, file_name: str) -> None:
        """
        最後に処理したファイル名をセッションに保存する

        Args:
            file_name: ファイル名
        """
        pass

    @abstractmethod
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        チャット履歴を取得する

        Returns:
            List[Dict[str, str]]: チャット履歴
        """
        pass

    @abstractmethod
    def add_chat_message(self, role: str, content: str) -> None:
        """
        チャットメッセージを追加する

        Args:
            role: メッセージの役割（"user" or "assistant"）
            content: メッセージの内容
        """
        pass

    @abstractmethod
    def is_new_file(self, file_name: str) -> bool:
        """
        新しいファイルかどうかを判定する

        Args:
            file_name: ファイル名

        Returns:
            bool: 新しいファイルかどうか
        """
        pass

    @classmethod
    @abstractmethod
    def create(cls) -> ISessionManager:
        """
        SessionManagerのインスタンスを作成する

        Returns:
            ISessionManager: 新しいSessionManagerインスタンス
        """
        pass
