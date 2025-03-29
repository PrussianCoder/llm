from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class ITextProcessor(ABC):
    """
    テキストの処理とレポート生成を行うためのインターフェース

    音声認識結果からのレポート生成、要約、議事録作成などの機能を定義します。
    """

    @abstractmethod
    def generate_report(
        self,
        transcript: str,
        model: str = "gpt-3.5-turbo",
        report_type: str = "要約",
        openai_api_key: Optional[str] = None,
    ) -> str:
        """
        テキスト文字起こしからレポートを生成する

        Args:
            transcript: 文字起こしテキスト
            model: OpenAIモデル名
            report_type: 生成するレポートタイプ
            openai_api_key: OpenAI APIキー

        Returns:
            str: 生成されたレポート
        """
        pass

    @abstractmethod
    def generate_chat_response(
        self,
        question: str,
        transcript: str,
        report: str,
        model: str = "gpt-3.5-turbo",
        openai_api_key: Optional[str] = None,
    ) -> str:
        """
        チャット質問に対する回答を生成する

        Args:
            question: ユーザーからの質問
            transcript: 文字起こしテキスト
            report: 生成されたレポート
            model: OpenAIモデル名
            openai_api_key: OpenAI APIキー

        Returns:
            str: 生成された回答
        """
        pass

    @classmethod
    @abstractmethod
    def create(cls) -> ITextProcessor:
        """
        TextProcessorのインスタンスを作成する

        Returns:
            ITextProcessor: 新しいTextProcessorインスタンス
        """
        pass
