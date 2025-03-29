from __future__ import annotations

import multiprocessing
from typing import Any, Dict

import streamlit as st
from interfaces.i_ui_component import ISidebarComponent
from settings.app_settings import AppSettings


class SidebarComponent(ISidebarComponent):
    """
    アプリケーションのサイドバーUIコンポーネント

    設定やオプション選択のUIを提供します。
    """

    def __init__(self) -> None:
        """SidebarComponentのインスタンスを初期化"""
        self.settings = AppSettings()

    def render(self) -> Dict[str, Any]:
        """
        UIコンポーネントをレンダリングする

        Returns:
            Dict[str, Any]: 設定値の辞書
        """
        return self.setup_sidebar()

    def setup_sidebar(self) -> Dict[str, Any]:
        """
        サイドバーのUI要素をセットアップする

        Returns:
            Dict[str, Any]: 設定値の辞書
        """
        st.sidebar.title("🛠️ 設定")

        # セッション状態の初期化
        if "settings" not in st.session_state:
            st.session_state.settings = {}

        # デフォルト設定から開始
        settings = self.settings.get_default_settings()

        with st.sidebar.expander("🧠 OpenAI API設定", expanded=True):
            # OpenAIのAPIキー設定
            if "openai_api_key" not in st.session_state:
                st.session_state.openai_api_key = ""

            api_key = st.text_input(
                "OpenAI APIキー",
                value=st.session_state.openai_api_key,
                type="password",
                help="OpenAIのAPIキーを入力してください。入力されたキーはセッション中のみ保存されます。",
                key="openai_api_key_input",
            )

            # APIキーをセッションステートに保存
            if api_key:
                st.session_state.openai_api_key = api_key

            # モデル選択
            settings["model"] = st.selectbox(
                "使用モデル",
                options=self.settings.openai_model_options,
                index=0,
                help="レポート生成に使用するモデルを選択してください。高品質なレポートにはGPT-4をお勧めします。",
                key="model",
            )

        # リアルタイム処理モード
        settings["realtime_mode"] = st.sidebar.checkbox(
            "リアルタイム処理モード",
            value=True,
            help="処理中にリアルタイムで結果を表示します",
        )

        with st.sidebar.expander("🎤 音声処理設定", expanded=True):
            # 言語設定
            selected_language = st.selectbox(
                "認識言語",
                options=list(self.settings.language_options.keys()),
                index=list(self.settings.language_options.keys()).index("英語（米国）"),
                help="音声の言語を選択してください。",
                key="language_selector",
            )
            settings["language"] = self.settings.language_options[selected_language]

            # 音声認識エンジン選択
            settings["recognition_engine"] = st.selectbox(
                "認識エンジン",
                options=self.settings.engine_options,
                index=0,  # Whisperをデフォルトに
                help="使用する音声認識エンジンを選択してください。Sphinxはオフラインで動作します。Whisperは高精度な音声認識が可能です。",
                key="recognition_engine",
            )

            # Whisper関連の設定（Whisperが選択されている場合のみ表示）
            if "Whisper" in settings["recognition_engine"]:
                st.info(
                    "💡 Whisperは高精度な音声認識が可能ですが、初回実行時にモデルをダウンロードする必要があります。"
                )

                # Whisperモデルのサイズ選択（高性能モデルが選択されている場合は非表示）
                if settings["recognition_engine"] == "Whisper":
                    settings["whisper_model_size"] = st.selectbox(
                        "Whisperモデルサイズ",
                        options=self.settings.whisper_model_options,
                        index=2,  # smallをデフォルトに
                        help="小さいモデルほど速く、大きいモデルほど精度が高くなります。",
                        key="whisper_model_size",
                    )

                # 言語検出オプション
                settings["whisper_detect_language"] = st.checkbox(
                    "言語を自動検出",
                    value=False,
                    help="チェックすると言語を自動検出します。精度向上のため、特定の言語がわかっている場合はオフにしてください。",
                    key="whisper_detect_language",
                )

            # ロングスピーチモード
            settings["long_speech_mode"] = st.checkbox(
                "ロングスピーチモード",
                value=True,
                help="長いスピーチやプレゼンテーションの音声認識に最適化します。",
                key="long_speech_mode",
            )

        with st.sidebar.expander("🔊 詳細音声設定", expanded=False):
            # ノイズ削減
            settings["reduce_noise"] = st.checkbox(
                "低音ノイズを削減",
                value=True,
                help="低周波ノイズを削減して音声認識の精度を向上させます。",
                key="reduce_noise",
            )

            # 無音部分の削除
            settings["remove_silence"] = st.checkbox(
                "無音部分を削除",
                value=True,
                help="長い無音部分を削除して認識精度を向上させます。",
                key="remove_silence",
            )

            # 音声強調
            settings["audio_enhancement"] = st.checkbox(
                "音声強調",
                value=True,
                help="音声を強調して認識精度を向上させます。",
                key="audio_enhancement",
            )

            # 認識試行回数
            settings["recognition_attempts"] = st.slider(
                "認識試行回数",
                min_value=1,
                max_value=5,
                value=3,
                help="音声認識の試行回数を設定します。回数を増やすと精度が向上する可能性がありますが、処理時間が長くなります。",
                key="recognition_attempts",
            )

            # 最小単語数
            settings["min_word_count"] = st.slider(
                "最小単語数",
                min_value=1,
                max_value=10,
                value=3,
                help="認識結果として採用する最小単語数を設定します。",
                key="min_word_count",
            )

            # チャンク分割時間
            settings["chunk_duration"] = st.slider(
                "チャンク分割時間（秒）",
                min_value=10,
                max_value=60,
                value=30,
                help="音声を分割する際の1チャンクあたりの秒数を設定します。",
                key="chunk_duration",
            )

            # 並列処理設定
            settings["parallel_processing"] = st.checkbox(
                "並列処理を有効化",
                value=True,
                help="複数のCPUコアを使用して並列処理を行います。処理速度が向上しますが、メモリ使用量が増加します。",
                key="parallel_processing",
            )

            # ワーカー数
            if settings["parallel_processing"]:
                max_cpu = multiprocessing.cpu_count()
                settings["max_workers"] = st.slider(
                    "並列ワーカー数",
                    min_value=2,
                    max_value=max_cpu,
                    value=min(4, max_cpu),
                    help="並列処理に使用するワーカー数を設定します。",
                    key="max_workers",
                )

        with st.sidebar.expander("📝 レポート設定", expanded=False):
            # レポートタイプ
            settings["report_type"] = st.selectbox(
                "レポートタイプ",
                options=self.settings.report_types,
                index=0,
                help="生成するレポートのタイプを選択してください。",
                key="report_type",
            )

        # 詳細時間範囲設定
        with st.sidebar.expander("⏱️ 処理時間範囲", expanded=False):
            # 処理開始時間（分）
            settings["start_minute"] = st.number_input(
                "処理開始時間（分）",
                min_value=0,
                value=0,
                help="処理を開始する時間（分）を指定します。0の場合は最初から処理します。",
                key="start_minute",
            )

            # 処理終了時間（分）
            settings["end_minute"] = st.number_input(
                "処理終了時間（分）",
                min_value=0,
                value=0,
                help="処理を終了する時間（分）を指定します。0の場合は最後まで処理します。",
                key="end_minute",
            )

        # セッション状態に設定を保存
        st.session_state.settings = settings

        return settings

    def get_api_key(self) -> str:
        """
        OpenAI APIキーを取得する

        Returns:
            str: APIキー
        """
        api_key = st.session_state.get("openai_api_key", "")
        if not api_key:
            st.warning("⚠️ サイドバーでOpenAI APIキーを設定してください")
        return api_key

    @classmethod
    def create(cls) -> ISidebarComponent:
        """
        SidebarComponentのインスタンスを作成する

        Returns:
            ISidebarComponent: 新しいSidebarComponentインスタンス
        """
        return SidebarComponent()
