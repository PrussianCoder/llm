from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydub import AudioSegment


class IUIComponent(ABC):
    """
    UIコンポーネントのベースインターフェース

    すべてのUIコンポーネントの基本機能を定義します。
    """

    @abstractmethod
    def render(self) -> Any:
        """
        UIコンポーネントをレンダリングする

        Returns:
            Any: レンダリング結果（コンポーネントによって異なる）
        """
        pass

    @classmethod
    @abstractmethod
    def create(cls) -> IUIComponent:
        """
        UIコンポーネントのインスタンスを作成する

        Returns:
            IUIComponent: 新しいUIコンポーネントインスタンス
        """
        pass


class ISidebarComponent(IUIComponent):
    """
    サイドバーコンポーネントのインターフェース

    設定やオプションの入力用UIを定義します。
    """

    @abstractmethod
    def setup_sidebar(self) -> Dict[str, Any]:
        """
        サイドバーのUI要素をセットアップする

        Returns:
            Dict[str, Any]: 設定値の辞書
        """
        pass

    @abstractmethod
    def get_api_key(self) -> str:
        """
        OpenAI APIキーを取得する

        Returns:
            str: APIキー
        """
        pass


class ITranscriptionComponent(IUIComponent):
    """
    文字起こし表示コンポーネントのインターフェース

    文字起こし結果の表示機能を定義します。
    """

    @abstractmethod
    def show_transcription_result(self, transcript: str) -> None:
        """
        文字起こし結果を表示する

        Args:
            transcript: 文字起こし結果のテキスト
        """
        pass

    @abstractmethod
    def show_chunk_details(self, chunks: List[AudioSegment], chunk_results: List[str]) -> None:
        """
        チャンク詳細を表示する

        Args:
            chunks: 音声チャンクのリスト
            chunk_results: チャンク毎の文字起こし結果のリスト
        """
        pass

    @abstractmethod
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
        pass


class IReportComponent(IUIComponent):
    """
    レポート表示コンポーネントのインターフェース

    生成されたレポートの表示機能を定義します。
    """

    @abstractmethod
    def show_report(self, report: str, report_type: str, file_name: str) -> None:
        """
        生成されたレポートを表示する

        Args:
            report: レポートのテキスト
            report_type: レポートタイプ
            file_name: 元のファイル名
        """
        pass


class IChatComponent(IUIComponent):
    """
    チャットコンポーネントのインターフェース

    チャット機能のUI要素を定義します。
    """

    @abstractmethod
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
        pass
