#!/usr/bin/env python3
from __future__ import annotations

"""
音声ファイル処理アプリケーション
"""
import os
import sys
import traceback
from typing import Callable, Dict, List

# インポートパスを修正
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# まず最初にStreamlitをインポート
import streamlit as st

# 最初のStreamlit命令としてページ設定を行う
st.set_page_config(page_title="会議記録レポート生成ツール", page_icon="🎙️", layout="wide")

# その他のimportはページ設定の後に行う
from dotenv import load_dotenv
from interfaces.i_audio_processor import IAudioProcessor
from services.session_manager import SessionManager
from services.text_service import TextService
from utils.error_handler import ErrorHandler
from utils.file_handler import FileHandler
from utils.logging_config import LoggingConfig

from ui.app_ui import AppUI

# FFmpegフォールバック機能のインポート
try:
    from utils.pydub_utils import setup_pydub_fallback

    # pydubのフォールバック機能を設定
    setup_pydub_fallback()
    print("pydubのフォールバック機能を設定しました。FFmpegがない環境でも基本機能が動作します。")
except Exception as e:
    print(f"pydubフォールバックの設定に失敗しました: {str(e)}")

# 環境変数のロード
load_dotenv()

# ロギングの設定
LoggingConfig.setup_basic_logging()
logger = LoggingConfig.get_logger("AudioReportApp")


class AudioReportApp:
    """
    音声/動画からレポートを生成するメインアプリケーション

    アプリケーションのコアロジックを管理し、UIとプロセッサー間の連携を行います。
    """

    def __init__(self, audio_processor: IAudioProcessor) -> None:
        """
        アプリケーションを初期化する

        Args:
            audio_processor: 音声処理インスタンス
        """
        self.ui = AppUI()
        self.audio_processor = audio_processor
        self.text_processor = TextService.create()
        self.session_manager = SessionManager.create()
        self.file_handler = FileHandler()
        self.error_handler = ErrorHandler()

    def process_question(self, question: str, api_key: str) -> None:
        """
        チャット質問を処理する

        Args:
            question: ユーザーからの質問
            api_key: OpenAI APIキー
        """
        # 音声認識結果の取得
        transcript = self.session_manager.get_transcript()

        # 生成されたレポートの取得（仮の実装、実際にはセッションから取得する必要あり）
        report_type = st.session_state.settings.get("report_type", "要約")
        report = self.text_processor.generate_report(
            transcript, report_type=report_type, openai_api_key=api_key
        )

        # チャット応答の生成
        answer = self.text_processor.generate_chat_response(
            question, transcript, report, openai_api_key=api_key
        )

        # チャット履歴に追加
        self.session_manager.add_chat_message("user", question)
        self.session_manager.add_chat_message("assistant", answer)

        # ページの再読み込み（チャット履歴を表示するため）
        st.rerun()

    def run(self) -> None:
        """アプリケーションを実行する"""
        try:
            # タイトルなどを設定（ページ設定はmain.pyの先頭で行う）
            st.title(f"🎙️ 会議記録レポート生成ツール")
            st.write(
                "MP3形式の会議録やMP4形式の動画をアップロードして、AI要約レポートを生成します。"
            )

            # サイドバーの設定（APIキー取得の前に実行）
            settings = self.ui.setup_sidebar()

            # OpenAI APIキーの入力と確認
            api_key = self.ui.get_api_key()

            # ファイルアップロード
            uploaded_file = self.ui.file_upload()

            # 新しいファイルがアップロードされたかどうかをチェック
            is_new_file = uploaded_file and self.session_manager.is_new_file(uploaded_file.name)

            # APIキーとファイルがある場合のみ処理
            if api_key and uploaded_file:
                # 新しいファイルがアップロードされた場合のみ音声処理を実行
                if is_new_file or self.session_manager.get_transcript() is None:
                    try:
                        # 一時ファイルを作成して保存
                        temp_file_path = self.file_handler.create_temp_file(uploaded_file)

                        try:
                            # ファイル処理開始を通知
                            st.info(f"'{uploaded_file.name}' の処理を開始します...")

                            # コールバック関数の設定（リアルタイムモード用）
                            def on_chunk_processed(
                                chunk_index: int,
                                total_chunks: int,
                                chunk_text: str,
                                chunk_duration: float,
                            ) -> None:
                                """
                                チャンク処理完了時に呼び出されるコールバック関数

                                Args:
                                    chunk_index: チャンクインデックス
                                    total_chunks: チャンク総数
                                    chunk_text: チャンクの認識結果
                                    chunk_duration: チャンクの長さ（秒）
                                """
                                # UIを更新
                                self.ui.update_realtime_results(
                                    chunk_text,
                                    chunk_index + 1,  # 1-indexedに変換
                                    total_chunks,
                                    chunk_duration,
                                )

                            # Whisperのデフォルト設定（smallモデル）
                            if "whisper_model_size" not in settings:
                                settings["whisper_model_size"] = "small"

                            # デフォルトのエンジンをWhisperに設定
                            if settings["recognition_engine"] == "Sphinx (推奨)":
                                settings["recognition_engine"] = "Whisper"

                            # 音声ファイルの処理と文字起こし
                            transcript, chunks, chunk_results = self.audio_processor.process_audio(
                                temp_file_path,
                                engine=settings["recognition_engine"],
                                language=settings["language"],
                                recognition_attempts=settings["recognition_attempts"],
                                reduce_noise=settings["reduce_noise"],
                                remove_silence=settings["remove_silence"],
                                audio_enhancement=settings["audio_enhancement"],
                                long_speech_mode=settings["long_speech_mode"],
                                chunk_duration=settings["chunk_duration"],
                                start_minute=settings["start_minute"],
                                end_minute=settings["end_minute"],
                                parallel_processing=settings["parallel_processing"],
                                max_workers=settings["max_workers"],
                                on_chunk_processed=(
                                    on_chunk_processed if settings["realtime_mode"] else None
                                ),
                                # Whisper関連のパラメータを追加
                                whisper_model_size=settings.get("whisper_model_size", "small"),
                                whisper_detect_language=settings.get(
                                    "whisper_detect_language", False
                                ),
                            )

                            # セッションステートに結果を保存
                            self.session_manager.set_transcript(transcript)
                            self.session_manager.set_chunks(chunks)
                            self.session_manager.set_chunk_results(chunk_results)
                            self.session_manager.set_last_file_name(uploaded_file.name)

                        finally:
                            # 一時ファイルの削除
                            self.file_handler.delete_file(temp_file_path)

                    except Exception as e:
                        # 処理中のエラーを表示
                        self.ui.show_error(
                            f"音声処理中にエラーが発生しました: {str(e)}", include_traceback=True
                        )
                        return

                # 既存の処理結果を使用
                transcript = self.session_manager.get_transcript()
                chunks = self.session_manager.get_chunks()
                chunk_results = self.session_manager.get_chunk_results()

                # リアルタイムモードでなければ、結果を一括表示
                if not settings["realtime_mode"]:
                    # 文字起こし結果の表示
                    st.subheader("音声認識結果")

                    # 2カラムで表示
                    col1, col2 = st.columns(2)

                    with col1:
                        # 文字起こし結果の表示
                        self.ui.show_transcription_result(transcript)

                    with col2:
                        # チャンク詳細の表示
                        self.ui.show_chunk_details(chunks, chunk_results)
                else:
                    # リアルタイムモードの場合は、処理完了後に最終結果を表示
                    self.ui.show_realtime_transcription(chunks, chunk_results)

                # 水平線を挿入
                st.markdown("---")

                # 文字起こしテキストが十分な長さの場合はレポート生成を行う
                if transcript and len(transcript.split()) > 20:
                    # レポート生成
                    report = self.text_processor.generate_report(
                        transcript,
                        report_type=settings["report_type"],
                        openai_api_key=api_key,
                    )

                    # レポート表示
                    self.ui.show_report(report, settings["report_type"], uploaded_file.name)

                    # ChatBoxの表示
                    st.markdown("---")
                    st.subheader("💬 内容について質問する")

                    # チャットインターフェースの表示
                    def on_question_submit(question: str) -> None:
                        self.process_question(question, api_key)

                    self.ui.show_chat_interface(
                        self.session_manager.get_chat_history(), on_question_submit
                    )

                elif transcript:
                    st.warning("文字起こしテキストが短すぎます。有効なレポートを生成できません。")
                else:
                    st.error("文字起こしに失敗しました。音声ファイルを確認してください。")

        except Exception as e:
            # エラーメッセージとスタックトレースの表示
            self.ui.show_error(f"エラーが発生しました: {str(e)}", include_traceback=True)


# 実行エントリポイント
if __name__ == "__main__":
    from services.audio_processor import (
        AudioProcessorV2,
    )  # 正しいインポートパスに修正

    app = AudioReportApp(AudioProcessorV2())
    app.run()
