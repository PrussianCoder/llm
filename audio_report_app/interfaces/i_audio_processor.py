from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

from pydub import AudioSegment


class IAudioProcessor(ABC):
    """
    音声処理とテキスト変換を行うためのインターフェース

    音声ファイルの処理、チャンク分割、音声認識などの機能を定義します。
    """

    @abstractmethod
    def process_audio(
        self,
        file_path: str,
        engine: str,
        language: str,
        reduce_noise: bool,
        remove_silence: bool,
        recognition_attempts: int,
        min_word_count: int,
        long_speech_mode: bool,
        chunk_duration: int,
        audio_enhancement: bool,
        start_minute: int,
        end_minute: int,
        parallel_processing: bool,
        max_workers: int,
        on_chunk_processed: Optional[Callable[[int, int, str, float], None]],
        whisper_model_size: str,
        whisper_detect_language: bool,
    ) -> Tuple[str, List[AudioSegment], List[str]]:
        """
        音声ファイルを処理して文字起こしを行う

        Args:
            file_path: 音声ファイルのパス
            engine: 使用する音声認識エンジン
            language: 認識する言語コード
            reduce_noise: ノイズ削減をするかどうか
            remove_silence: 無音部分を削除するかどうか
            recognition_attempts: 認識試行回数
            min_word_count: 最小単語数
            long_speech_mode: 長いスピーチモードを使用するかどうか
            chunk_duration: チャンク分割時間（秒）
            audio_enhancement: 音声強調をするかどうか
            start_minute: 処理開始時間（分）
            end_minute: 処理終了時間（分）
            parallel_processing: 並列処理をするかどうか
            max_workers: 並列処理時のワーカー数
            on_chunk_processed: チャンク処理完了時のコールバック関数
            whisper_model_size: Whisperモデルのサイズ (tiny/base/small/medium/large)
            whisper_detect_language: 言語を自動検出するかどうか

        Returns:
            Tuple[str, List[AudioSegment], List[str]]: (transcript, chunks, chunk_results) - 文字起こし結果、チャンクリスト、チャンク結果リスト
        """
        pass

    @abstractmethod
    def preprocess_audio(
        self,
        audio_data: AudioSegment,
        reduce_noise: bool,
        remove_silence: bool,
        audio_enhancement: bool,
    ) -> AudioSegment:
        """
        音声データを前処理する

        Args:
            audio_data: 処理する音声データ
            reduce_noise: ノイズ削減をするかどうか
            remove_silence: 無音部分を削除するかどうか
            audio_enhancement: 音声強調をするかどうか

        Returns:
            AudioSegment: 前処理された音声データ
        """
        pass

    @abstractmethod
    def split_audio_into_chunks(
        self,
        audio_data: AudioSegment,
        min_silence_len: int,
        silence_thresh: int,
        max_chunk_duration: int,
        long_speech_mode: bool,
    ) -> List[AudioSegment]:
        """
        音声データをチャンクに分割する

        Args:
            audio_data: 分割する音声データ
            min_silence_len: 最小無音長（ミリ秒）
            silence_thresh: 無音判定閾値（dB）
            max_chunk_duration: 最大チャンク時間（ミリ秒）
            long_speech_mode: 長いスピーチモードを使用するかどうか

        Returns:
            List[AudioSegment]: 分割された音声チャンクのリスト
        """
        pass

    @classmethod
    @abstractmethod
    def create(cls) -> IAudioProcessor:
        """
        AudioProcessorのインスタンスを作成する

        Returns:
            IAudioProcessor: 新しいAudioProcessorインスタンス
        """
        pass
