from __future__ import annotations

from typing import Any, List

import pandas as pd
import streamlit as st
from interfaces.i_ui_component import ITranscriptionComponent
from pydub import AudioSegment


class TranscriptionComponent(ITranscriptionComponent):
    """
    文字起こし結果表示用のUIコンポーネント

    文字起こし結果の表示、チャンク詳細の表示、リアルタイム結果の表示などの機能を提供します。
    """

    def __init__(self) -> None:
        """TranscriptionComponentのインスタンスを初期化"""
        pass

    def render(self) -> Any:
        """
        UIコンポーネントをレンダリングする

        Returns:
            Any: レンダリング結果
        """
        st.subheader("音声認識結果")
        return None

    def show_transcription_result(self, transcript: str) -> None:
        """
        文字起こし結果を表示する

        Args:
            transcript: 文字起こし結果のテキスト
        """
        if not transcript:
            st.warning("文字起こしが空です。音声認識に問題がある可能性があります。")
            return

        # 文字起こし結果を表示
        st.write("#### 文字起こし全文")
        st.text_area(
            "",
            transcript,
            height=200,
            key="transcription_result_display",
        )

    def show_chunk_details(self, chunks: List[AudioSegment], chunk_results: List[str]) -> None:
        """
        チャンク詳細を表示する

        Args:
            chunks: 音声チャンクのリスト
            chunk_results: チャンク毎の文字起こし結果のリスト
        """
        st.write("#### チャンク詳細")

        # チャンクがない場合
        if not chunks or not chunk_results:
            st.info("チャンク情報がありません。")
            return

        # データフレームの作成
        chunk_data = []
        for i, (chunk, result) in enumerate(zip(chunks, chunk_results)):
            chunk_data.append(
                {
                    "チャンク": i + 1,
                    "長さ(秒)": round(len(chunk) / 1000, 1),
                    "単語数": len(result.split()) if result else 0,
                    "認識テキスト": result[:100] + ("..." if len(result) > 100 else ""),
                }
            )

        # データフレームの表示
        if chunk_data:
            df = pd.DataFrame(chunk_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("チャンク情報がありません。")

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
        # リアルタイム表示用のコンテナが初期化されていることを確認
        if "realtime_text_container" not in st.session_state:
            st.session_state.realtime_text_container = st.container()

        # プログレスバーの更新
        progress_container = st.empty()
        progress_container.progress(chunk_index / total_chunks)

        # ステータス表示の更新
        status_container = st.empty()
        status_container.info(
            f"チャンク {chunk_index}/{total_chunks} を処理中... "
            f"(長さ: {chunk_duration:.1f}秒, "
            f"単語数: {len(chunk_text.split()) if chunk_text else 0})"
        )

        # リアルタイムテキスト表示の更新
        with st.session_state.realtime_text_container:
            # テキストが空でなければ表示
            if chunk_text:
                st.text_area(
                    f"チャンク {chunk_index} の認識結果:",
                    chunk_text,
                    height=100,
                    key=f"realtime_text_{chunk_index}",
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
        st.subheader("音声認識結果（リアルタイム処理完了）")

        # 全文を表示
        full_transcript = " ".join([result for result in chunk_results if result]).strip()
        if full_transcript:
            st.write("#### 文字起こし全文")
            st.text_area(
                "",
                full_transcript,
                height=200,
                key="realtime_full_transcript",
            )
        else:
            st.warning("文字起こし結果が空です。音声認識に問題がある可能性があります。")

        # チャンク詳細を表示
        self.show_chunk_details(chunks, chunk_results)

    @classmethod
    def create(cls) -> ITranscriptionComponent:
        """
        TranscriptionComponentのインスタンスを作成する

        Returns:
            ITranscriptionComponent: 新しいTranscriptionComponentインスタンス
        """
        return TranscriptionComponent()
