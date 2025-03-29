from __future__ import annotations

from typing import Any, Dict, List


class AppSettings:
    """
    アプリケーション設定を管理するクラス

    デフォルト設定や設定値の取得・設定を提供します。
    """

    def __init__(self) -> None:
        """AppSettingsのインスタンスを初期化"""
        # 音声処理に関するデフォルト設定
        self.default_audio_settings: Dict[str, Any] = {
            "reduce_noise": True,
            "remove_silence": True,
            "recognition_attempts": 3,
            "min_word_count": 3,
            "long_speech_mode": True,
            "chunk_duration": 30,
            "audio_enhancement": True,
            "start_minute": 0,
            "end_minute": 0,
            "parallel_processing": True,
            "max_workers": 4,
            "whisper_model_size": "small",
            "whisper_detect_language": False,
        }

        # UI関連のデフォルト設定
        self.default_ui_settings: Dict[str, Any] = {
            "realtime_mode": True,
        }

        # レポート関連のデフォルト設定
        self.default_report_settings: Dict[str, Any] = {
            "report_type": "会議録",
            "model": "gpt-3.5-turbo",
        }

        # 言語オプション
        self.language_options: Dict[str, str] = {
            "英語（米国）": "en",
            "日本語": "ja",
            "中国語": "zh",
            "フランス語": "fr",
            "ドイツ語": "de",
            "イタリア語": "it",
            "スペイン語": "es",
            "韓国語": "ko",
        }

        # 音声認識エンジンオプション
        self.engine_options: List[str] = [
            "Whisper",
            "Whisper (高性能)",
            "Faster Whisper",
            "Sphinx (推奨)",
            "完全音声処理 (Sphinxベース)",
            "Multiple Engines",
            "Google Speech Recognition",
        ]

        # Whisperモデルサイズオプション
        self.whisper_model_options: List[str] = ["tiny", "base", "small", "medium", "large"]

        # レポートタイプオプション
        self.report_types: List[str] = ["会議録", "議事録", "要約", "アクションアイテム", "Q&A抽出"]

        # OpenAIモデルオプション
        self.openai_model_options: List[str] = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
        ]

    def get_default_settings(self) -> Dict[str, Any]:
        """
        すべてのデフォルト設定を取得する

        Returns:
            Dict[str, Any]: デフォルト設定の辞書
        """
        settings = {}
        settings.update(self.default_audio_settings)
        settings.update(self.default_ui_settings)
        settings.update(self.default_report_settings)
        # 言語設定のデフォルト
        settings["language"] = self.language_options["英語（米国）"]
        # 音声認識エンジンのデフォルト（Whisper）
        settings["recognition_engine"] = self.engine_options[0]
        return settings
