from __future__ import annotations

import os
import re
import subprocess
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

import speech_recognition as sr
from interfaces.i_audio_processor import IAudioProcessor
from pydub import AudioSegment
from pydub.silence import split_on_silence
from utils.logging_config import LoggingConfig

# ロガーの取得
logger = LoggingConfig.get_logger("AudioProcessor")


class AudioProcessorV2(IAudioProcessor):
    """音声処理とテキスト変換を行うクラス"""

    def __init__(self) -> None:
        """AudioProcessorの初期化"""
        self.recognizer = sr.Recognizer()
        self._audio_cache = {}
        self._recognition_cache = {}

    @classmethod
    def create(cls) -> IAudioProcessor:
        """
        AudioProcessorのインスタンスを作成する

        Returns:
            IAudioProcessor: 新しいAudioProcessorインスタンス
        """
        return cls()

    def process_audio(
        self,
        file_path: str,
        engine: str = "Whisper",
        language: str = "ja-JP",
        reduce_noise: bool = True,
        remove_silence: bool = True,
        recognition_attempts: int = 3,
        min_word_count: int = 3,
        long_speech_mode: bool = True,
        chunk_duration: int = 30,
        audio_enhancement: bool = True,
        start_minute: int = 0,
        end_minute: int = 0,
        parallel_processing: bool = True,
        max_workers: int = 4,
        on_chunk_processed: Optional[Callable[[int, int, str, float], None]] = None,
        whisper_model_size: str = "base",
        whisper_detect_language: bool = False,
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
            whisper_model_size: Whisperモデルのサイズ
            whisper_detect_language: 言語を自動検出するかどうか

        Returns:
            Tuple[str, List[AudioSegment], List[str]]: (transcript, chunks, chunk_results) - 文字起こし結果、チャンクリスト、チャンク結果リスト
        """
        try:
            logger.info(f"音声ファイルの処理を開始します: {file_path}")
            logger.info(f"使用するエンジン: {engine}, 言語: {language}")

            # ファイルの存在チェック
            if not os.path.exists(file_path):
                logger.error(f"ファイルが存在しません: {file_path}")
                return "", [], []

            # FFmpegの利用可能性をチェック
            ffmpeg_available = self._check_ffmpeg_available()
            if not ffmpeg_available:
                logger.warning(
                    "FFmpegが利用できないため、一部の機能が制限されます。mp3/mp4ファイルの処理やノイズ削減が制限される可能性があります。"
                )
                logger.warning("これはStreamlit Cloudでの制限である可能性があります。")

                # 可能であれば.wavファイルを使用することを推奨
                if os.path.splitext(file_path)[1].lower() not in [".wav"]:
                    logger.warning(
                        "FFmpegがない環境では.wavファイルの使用を推奨します。mp3/mp4ファイルの処理が失敗する可能性があります。"
                    )

            # 音声ファイルの読み込み
            try:
                logger.info(f"音声ファイルを読み込み中: {file_path}")
                audio_data = self._load_audio_file(file_path, start_minute, end_minute)
                logger.info(
                    f"音声ファイルの読み込みに成功しました: 長さ {len(audio_data)/1000:.2f}秒"
                )
            except Exception as e:
                logger.error(f"音声ファイルの読み込みに失敗しました: {str(e)}")
                return "", [], []

            # 音声の前処理
            try:
                logger.info("音声の前処理を開始します...")
                preprocessed_audio = self.preprocess_audio(
                    audio_data, reduce_noise, remove_silence, audio_enhancement
                )
                logger.info(
                    f"音声の前処理が完了しました: 処理後の長さ {len(preprocessed_audio)/1000:.2f}秒"
                )
            except Exception as e:
                logger.error(f"音声の前処理に失敗しました: {str(e)}")
                # 前処理に失敗した場合は元の音声データを使用
                preprocessed_audio = audio_data
                logger.info("前処理をスキップし、オリジナルの音声を使用します")

            # チャンクへの分割
            try:
                logger.info("音声のチャンク分割を開始します...")
                chunks = self.split_audio_into_chunks(
                    preprocessed_audio,
                    max_chunk_duration=chunk_duration * 1000,  # 秒からミリ秒に変換
                    long_speech_mode=long_speech_mode,
                )
                logger.info(f"音声を {len(chunks)} チャンクに分割しました")
            except Exception as e:
                logger.error(f"音声のチャンク分割に失敗しました: {str(e)}")
                # チャンク分割に失敗した場合は単一チャンクとして扱う
                chunks = [preprocessed_audio]
                logger.info("チャンク分割をスキップし、単一チャンクとして処理します")

            # 各チャンクを認識
            chunk_results = []

            # 処理するチャンクのインデックスリスト
            chunk_indices = list(range(len(chunks)))

            # 並列処理の設定
            if parallel_processing and len(chunk_indices) > 1:
                logger.info(f"並列処理を開始します（ワーカー数: {max_workers}）")

                # 並列処理用の関数
                def process_chunk(chunk_idx):
                    try:
                        chunk = chunks[chunk_idx]
                        logger.info(
                            f"チャンク {chunk_idx+1}/{len(chunks)} の処理を開始します（長さ: {len(chunk)/1000:.2f}秒）"
                        )

                        # チャンク認識
                        result = self._recognize_chunk(
                            chunk,
                            engine=engine,
                            language=language,
                            recognition_attempts=recognition_attempts,
                            min_word_count=min_word_count,
                            whisper_model_size=whisper_model_size,
                            whisper_detect_language=whisper_detect_language,
                        )

                        # コールバック関数があれば呼び出し（リアルタイムモード用）
                        if on_chunk_processed:
                            on_chunk_processed(chunk_idx, len(chunks), result, len(chunk) / 1000)

                        logger.info(
                            f"チャンク {chunk_idx+1}/{len(chunks)} の処理が完了しました: {len(result.split())} 単語認識"
                        )
                        return result
                    except Exception as e:
                        logger.error(
                            f"チャンク {chunk_idx+1}/{len(chunks)} の処理に失敗しました: {str(e)}"
                        )
                        return ""

                # 並列処理実行
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    chunk_results = list(executor.map(process_chunk, chunk_indices))
            else:
                logger.info("逐次処理を開始します")
                for idx, chunk in enumerate(chunks):
                    try:
                        logger.info(
                            f"チャンク {idx+1}/{len(chunks)} の処理を開始します（長さ: {len(chunk)/1000:.2f}秒）"
                        )

                        # チャンク認識
                        result = self._recognize_chunk(
                            chunk,
                            engine=engine,
                            language=language,
                            recognition_attempts=recognition_attempts,
                            min_word_count=min_word_count,
                            whisper_model_size=whisper_model_size,
                            whisper_detect_language=whisper_detect_language,
                        )

                        # コールバック関数があれば呼び出し（リアルタイムモード用）
                        if on_chunk_processed:
                            on_chunk_processed(idx, len(chunks), result, len(chunk) / 1000)

                        chunk_results.append(result)
                        logger.info(
                            f"チャンク {idx+1}/{len(chunks)} の処理が完了しました: {len(result.split())} 単語認識"
                        )
                    except Exception as e:
                        logger.error(
                            f"チャンク {idx+1}/{len(chunks)} の処理に失敗しました: {str(e)}"
                        )
                        chunk_results.append("")

            # 全チャンク結果を結合
            transcript = " ".join(chunk_results)
            logger.info(
                f"全チャンクの処理が完了しました。合計文字数: {len(transcript)}, 単語数: {len(transcript.split())}"
            )

            # 重複パターンの修正
            transcript = self._fix_repetition_patterns(transcript)
            logger.info(
                f"重複パターン修正後 - 合計文字数: {len(transcript)}, 単語数: {len(transcript.split())}"
            )

            # 空の文字起こし結果の場合はエラーメッセージを返す
            if not transcript.strip():
                logger.error("全てのチャンクの文字起こしに失敗しました。結果が空です。")
                transcript = "音声認識に失敗しました。ファイル形式や音声品質を確認してください。"

            logger.info("音声処理が完了しました")
            return transcript, chunks, chunk_results

        except Exception as e:
            logger.error(f"音声処理中にエラーが発生しました: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _recognize_chunk(
        self,
        chunk: AudioSegment,
        engine: str = "Whisper",
        language: str = "ja-JP",
        recognition_attempts: int = 3,
        min_word_count: int = 3,
        whisper_model_size: str = "base",
        whisper_detect_language: bool = False,
    ) -> str:
        """
        音声チャンクを認識する

        Args:
            chunk: 認識する音声チャンク
            engine: 使用する音声認識エンジン
            language: 認識する言語コード
            recognition_attempts: 認識試行回数
            min_word_count: 最小単語数
            whisper_model_size: Whisperモデルのサイズ
            whisper_detect_language: 言語を自動検出するかどうか

        Returns:
            str: 認識結果テキスト
        """
        logger = LoggingConfig.get_logger(self.__class__.__name__)
        logger.info(f"チャンクの認識を開始: エンジン={engine}, 言語={language}")

        # 認識結果
        best_result = ""
        best_word_count = 0

        # 短いチャンクは処理しない（無音など）
        if len(chunk) < 500:  # 500ミリ秒未満
            logger.warning(f"チャンクが短すぎるためスキップ: {len(chunk)}ms")
            return ""

        # 言語コードの調整（Whisperは2文字コード、GoogleとSphinxは地域コードあり）
        lang_code = language
        if engine.lower().startswith("whisper"):
            # Whisper用の言語コード調整
            if "-" in language:
                lang_code = language.split("-")[0]
            logger.info(f"Whisperに適した言語コードに調整: {language} -> {lang_code}")

        # 選択されたエンジンで認識
        for attempt in range(recognition_attempts):
            try:
                logger.info(f"認識試行 {attempt + 1}/{recognition_attempts}")

                if engine == "Sphinx":
                    result = self._recognize_sphinx(chunk, language)
                elif engine == "Google":
                    result = self._recognize_google(chunk, language)
                elif engine == "Whisper":
                    result = self._recognize_with_whisper(
                        chunk, lang_code, whisper_model_size, whisper_detect_language
                    )
                elif engine == "FasterWhisper":
                    result = self._recognize_with_faster_whisper(
                        chunk, lang_code, whisper_model_size, whisper_detect_language
                    )
                else:
                    # デフォルトのエンジンとしてWhisperを使用
                    logger.warning(f"未知のエンジン: {engine}, Whisperを使用します")
                    result = self._recognize_with_whisper(
                        chunk, lang_code, whisper_model_size, whisper_detect_language
                    )

                # 認識結果を整形（余分な空白の削除など）
                result = self._format_recognition_result(result)

                # 単語数をカウント
                word_count = len(result.split())

                logger.info(
                    f"認識結果: {word_count}単語 ('{result[:50]}{'...' if len(result) > 50 else ''}')"
                )

                # より多くの単語を含む結果を保持
                if word_count > best_word_count:
                    best_result = result
                    best_word_count = word_count
                    logger.info(f"より良い結果を更新: {best_word_count}単語")

                # 十分な単語数があれば早期終了
                if word_count >= min_word_count:
                    logger.info(f"十分な単語数 ({word_count} >= {min_word_count}) のため認識を終了")
                    break

            except Exception as e:
                logger.error(f"認識試行 {attempt + 1} で例外が発生: {str(e)}")
                # 続行して他の試行を実行

        # 最終的な結果を返す
        if best_word_count > 0:
            logger.info(f"チャンク認識完了: {best_word_count}単語")
            return best_result
        else:
            logger.warning("チャンク認識に失敗: 有効な結果なし")
            return ""

    def _fix_repetition_patterns(self, text: str) -> str:
        """
        テキスト内の繰り返しパターンを検出して修正する

        Args:
            text: 修正するテキスト

        Returns:
            str: 修正されたテキスト
        """
        if not text:
            return text

        # 最小フレーズ長（単語数）
        min_phrase_length = 5
        # 繰り返し閾値
        repeat_threshold = 3

        # 単語に分割
        words = text.split()
        if len(words) < min_phrase_length * 2:
            return text

        # フレーズの繰り返しを検出
        for phrase_len in range(min_phrase_length, min(20, len(words) // 2)):
            for start_idx in range(len(words) - phrase_len * 2):
                phrase = " ".join(words[start_idx : start_idx + phrase_len])
                count = 0

                # このフレーズが何回繰り返されているか確認
                current_idx = start_idx
                while current_idx <= len(words) - phrase_len:
                    current_phrase = " ".join(words[current_idx : current_idx + phrase_len])
                    if current_phrase == phrase:
                        count += 1
                        current_idx += phrase_len
                    else:
                        break

                # 閾値以上の繰り返しを検出した場合、ログに記録
                if count >= repeat_threshold:
                    logger.warning(
                        f"繰り返しパターンを検出: '{phrase}' が {count} 回繰り返されています"
                    )
                    # フレーズの繰り返しを1回だけにする新しいテキストを作成
                    new_words = words.copy()
                    new_words[start_idx : start_idx + phrase_len * count] = words[
                        start_idx : start_idx + phrase_len
                    ]
                    return " ".join(new_words)

        # 単純な単語の繰り返しに対する処理
        for i in range(len(words) - 1):
            # 同じ単語が3回以上連続で繰り返される場合に修正
            if words[i] == words[i + 1]:
                repeat_count = 1
                j = i
                while j < len(words) - 1 and words[j] == words[j + 1]:
                    repeat_count += 1
                    j += 1

                if repeat_count >= 3:
                    logger.warning(
                        f"単語の繰り返しを検出: '{words[i]}' が {repeat_count} 回繰り返されています"
                    )
                    # 3回以上連続する場合は2回だけにする
                    new_words = words.copy()
                    new_words[i : i + repeat_count] = [words[i], words[i]]
                    return " ".join(new_words)

        # 特定の繰り返しパターンに対する検出と修正
        repetition_patterns = [r"((?:\S+\s+){3,6}?)\1{2,}"]  # 3～6単語のフレーズが2回以上繰り返し

        for pattern in repetition_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                repeated_phrase = match.group(1).strip()
                full_match = match.group(0)
                if len(repeated_phrase) > 0:
                    logger.warning(f"正規表現で繰り返しパターンを検出: '{repeated_phrase}'")
                    # 繰り返しを1回だけに置換
                    text = text.replace(full_match, repeated_phrase)

        return text

    def _load_audio_file(
        self, file_path: str, start_minute: int = 0, end_minute: int = 0
    ) -> AudioSegment:
        """
        音声/動画ファイルを読み込む

        Args:
            file_path: ファイルパス
            start_minute: 開始時間（分）
            end_minute: 終了時間（分）（0の場合は最後まで）

        Returns:
            AudioSegment: 読み込まれた音声データ
        """
        logger = LoggingConfig.get_logger(self.__class__.__name__)
        logger.info(f"ファイルを読み込み中: {os.path.basename(file_path)}")

        # ファイル拡張子を取得
        ext = os.path.splitext(file_path)[1].lower()

        # MP4やその他の動画ファイルの場合、音声を抽出
        if ext in [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]:
            # FFmpegが利用可能かチェック
            ffmpeg_available = self._check_ffmpeg_available()

            if ffmpeg_available:
                # FFmpegを使用して音声を抽出
                temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                temp_mp3.close()

                try:
                    # FFmpegを使用してMP4からMP3を抽出
                    os.system(f'ffmpeg -i "{file_path}" -q:a 0 -map a "{temp_mp3.name}" -y')
                    audio_data = AudioSegment.from_mp3(temp_mp3.name)
                except Exception as e:
                    logger.error(f"FFmpegでの音声抽出に失敗しました: {str(e)}")
                    # 直接ファイルを開こうとする
                    try:
                        audio_data = AudioSegment.from_file(file_path)
                        logger.info("代替方法でファイルを読み込みました")
                    except Exception as e2:
                        logger.error(f"ファイルの読み込みに失敗しました: {str(e2)}")
                        raise ValueError(
                            f"ファイル '{os.path.basename(file_path)}' の読み込みに失敗しました。サポートされている形式かご確認ください。"
                        )
                finally:
                    # 一時ファイルを削除
                    if os.path.exists(temp_mp3.name):
                        os.unlink(temp_mp3.name)
            else:
                # FFmpegが利用できない場合は直接ファイルを開こうとする
                logger.warning("FFmpegが利用できないため、直接ファイルを読み込みます")
                try:
                    audio_data = AudioSegment.from_file(file_path)
                    logger.info("代替方法でファイルを読み込みました")
                except Exception as e:
                    logger.error(f"ファイルの読み込みに失敗しました: {str(e)}")
                    raise ValueError(
                        f"ファイル '{os.path.basename(file_path)}' の読み込みに失敗しました。サポートされている形式かご確認ください。FFmpegをインストールするとより多くの形式をサポートできます。"
                    )

        elif ext == ".mp3":
            try:
                audio_data = AudioSegment.from_mp3(file_path)
            except Exception as e:
                logger.error(f"MP3ファイルの読み込みに失敗しました: {str(e)}")
                raise ValueError(
                    f"MP3ファイル '{os.path.basename(file_path)}' の読み込みに失敗しました。FFmpegがインストールされているか確認してください。"
                )
        else:
            # その他の形式（wav等）
            try:
                audio_data = AudioSegment.from_file(file_path)
            except Exception as e:
                logger.error(f"音声ファイルの読み込みに失敗しました: {str(e)}")
                raise ValueError(
                    f"ファイル '{os.path.basename(file_path)}' の読み込みに失敗しました。サポートされている形式かご確認ください。"
                )

        # 時間範囲の指定がある場合は切り出し
        if start_minute > 0 or (end_minute > 0 and end_minute > start_minute):
            # 分をミリ秒に変換
            start_ms = start_minute * 60 * 1000

            if end_minute > 0:
                end_ms = end_minute * 60 * 1000
                # ファイルの長さよりも長い場合は調整
                if end_ms > len(audio_data):
                    end_ms = len(audio_data)

                logger.info(f"音声/動画から時間範囲を抽出: {start_minute}分から{end_minute}分まで")
                # 指定範囲を切り出し
                audio_data = audio_data[start_ms:end_ms]
            else:
                # 終了時間指定なしの場合は開始時間から最後まで
                logger.info(f"音声/動画から時間範囲を抽出: {start_minute}分から最後まで")
                audio_data = audio_data[start_ms:]

        return audio_data

    def _check_ffmpeg_available(self) -> bool:
        """
        FFmpegが利用可能かどうかを確認する

        Returns:
            bool: FFmpegが利用可能な場合はTrue、そうでない場合はFalse
        """
        logger = LoggingConfig.get_logger(self.__class__.__name__)
        try:
            # FFmpegが存在するかチェック
            subprocess.run(
                ["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )
            logger.info("FFmpegが利用可能です")
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpegが見つかりません。一部の機能が制限されます。")
            return False

    def preprocess_audio(
        self,
        audio_data: AudioSegment,
        reduce_noise: bool = True,
        remove_silence: bool = True,
        audio_enhancement: bool = True,
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
        logger = LoggingConfig.get_logger(self.__class__.__name__)
        logger.info("音声前処理を開始します")

        # サンプルレートを16kHzに変更（音声認識に最適）
        if audio_data.frame_rate != 16000:
            audio_data = audio_data.set_frame_rate(16000)

        # ステレオからモノラルに変換
        if audio_data.channels > 1:
            audio_data = audio_data.set_channels(1)

        # 音量正規化（音声認識のために適切なレベルに調整）
        if audio_enhancement:
            # RMSレベルを測定
            rms = audio_data.rms
            target_dBFS = -20  # 目標dBFS

            # 音量が小さすぎる場合は増幅
            if rms < 100:  # 非常に静かな音声の場合
                gain_needed = max(20, target_dBFS - audio_data.dBFS)
                audio_data = audio_data.apply_gain(gain_needed)
            else:
                # 標準的な正規化
                change_in_dBFS = target_dBFS - audio_data.dBFS
                audio_data = audio_data.apply_gain(change_in_dBFS)

        # 低周波ノイズの削減（ハイパスフィルター）
        if reduce_noise:
            # FFmpegが利用可能かチェック
            ffmpeg_available = self._check_ffmpeg_available()

            if ffmpeg_available:
                # pydubでハイパスフィルターを直接適用する方法がないため、
                # FFmpeg呼び出しを使用する代替手段として処理
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
                        temp_in_name = temp_in.name
                        temp_out_name = temp_out.name

                try:
                    # 一時ファイルに書き出し
                    audio_data.export(temp_in_name, format="wav")

                    # FFmpegで100Hz以下の周波数をカット（ハイパスフィルター）
                    os.system(
                        f'ffmpeg -i "{temp_in_name}" -af "highpass=f=100" "{temp_out_name}" -y'
                    )

                    # 処理された音声を読み込み
                    audio_data = AudioSegment.from_wav(temp_out_name)
                finally:
                    # 一時ファイルを削除
                    if os.path.exists(temp_in_name):
                        os.unlink(temp_in_name)
                    if os.path.exists(temp_out_name):
                        os.unlink(temp_out_name)
            else:
                logger.warning("FFmpegが利用できないため、ノイズ削減をスキップします")

        # 先頭と末尾の無音部分をトリミング
        if remove_silence:
            silence_threshold = -40  # dB
            audio_data = self._trim_silence(audio_data, silence_threshold)

        logger.info("音声前処理が完了しました")
        return audio_data

    def _trim_silence(
        self, audio_segment: AudioSegment, silence_threshold: int = -40, chunk_size: int = 10
    ) -> AudioSegment:
        """
        音声の先頭と末尾の無音部分をトリミングする

        Args:
            audio_segment: 処理する音声セグメント
            silence_threshold: 無音と判断するしきい値（dB）
            chunk_size: 検出するチャンクサイズ（ミリ秒）

        Returns:
            AudioSegment: トリミングされた音声セグメント
        """

        def detect_leading_silence(
            sound: AudioSegment, silence_threshold: int, chunk_size: int
        ) -> int:
            """
            音声の先頭の無音長を検出する
            """
            trim_ms = 0
            while trim_ms < len(sound):
                if sound[trim_ms : trim_ms + chunk_size].dBFS < silence_threshold:
                    trim_ms += chunk_size
                else:
                    break
            return trim_ms

        # 先頭の無音をトリミング
        start_trim = detect_leading_silence(audio_segment, silence_threshold, chunk_size)

        # 末尾の無音をトリミング（音声を逆にして先頭の無音を検出）
        end_trim = detect_leading_silence(audio_segment.reverse(), silence_threshold, chunk_size)

        # トリミングされた部分を返す
        duration = len(audio_segment)
        trimmed_audio = audio_segment[start_trim : duration - end_trim]

        return trimmed_audio

    def split_audio_into_chunks(
        self,
        audio_data: AudioSegment,
        min_silence_len: int = 500,
        silence_thresh: int = -40,
        max_chunk_duration: int = 30000,
        long_speech_mode: bool = True,
        overlap_percentage: int = 10,
        min_chunk_length: int = 2000,
    ) -> List[AudioSegment]:
        """
        音声データをチャンクに分割する

        Args:
            audio_data: 分割する音声データ
            min_silence_len: 最小無音長（ミリ秒）
            silence_thresh: 無音判定閾値（dB）
            max_chunk_duration: 最大チャンク時間（ミリ秒）
            long_speech_mode: 長いスピーチモードを使用するかどうか
            overlap_percentage: チャンク間のオーバーラップ割合（%）
            min_chunk_length: 最小チャンク長（ミリ秒）

        Returns:
            List[AudioSegment]: 分割された音声チャンクのリスト
        """
        logger.info("音声のチャンク分割を開始します")

        # チャンクリスト
        chunks = []

        try:
            # 音声の長さを取得
            audio_length_ms = len(audio_data)
            logger.info(f"音声の総時間: {audio_length_ms / 1000:.1f}秒")

            # 音声が短い場合はそのまま1つのチャンクとして返す
            if audio_length_ms <= max_chunk_duration:
                logger.info("音声が短いため、単一チャンクとして処理します")
                chunks = [audio_data]
                return chunks

            # 長いスピーチモードの場合、無音検出よりも固定長分割を優先
            if long_speech_mode:
                logger.info("長いスピーチモードで固定長分割を使用します")

                # オーバーラップの長さを計算（ミリ秒）
                overlap_ms = (max_chunk_duration * overlap_percentage) // 100

                # 1チャンクの長さ（オーバーラップを考慮）
                chunk_length_ms = max_chunk_duration - overlap_ms

                # オーバーラップ付きの分割位置を計算
                start_positions = list(range(0, audio_length_ms, chunk_length_ms))

                # 各開始位置からチャンクを生成
                for i, start_ms in enumerate(start_positions):
                    # チャンクの終了位置を計算（音声の最後を超えないように）
                    end_ms = min(start_ms + max_chunk_duration, audio_length_ms)

                    # チャンクを抽出
                    chunk = audio_data[start_ms:end_ms]

                    # チャンクの長さが最小要件を満たしていることを確認
                    if len(chunk) >= min_chunk_length:
                        chunks.append(chunk)
                        logger.info(
                            f"チャンク {i + 1}/{len(start_positions)} 作成: {len(chunk) / 1000:.1f}秒 ({start_ms / 1000:.1f}〜{end_ms / 1000:.1f}秒)"
                        )
            else:
                # 無音検出による分割
                logger.info("無音検出によるチャンク分割を使用します")

                # 無音検出によるチャンク分割
                chunks = split_on_silence(
                    audio_data,
                    min_silence_len=min_silence_len,
                    silence_thresh=silence_thresh,
                    keep_silence=100,  # 無音部分を少し残す
                )

                # 長すぎるチャンクを再分割
                final_chunks = []
                for i, chunk in enumerate(chunks):
                    if len(chunk) > max_chunk_duration:
                        logger.info(
                            f"チャンク {i + 1} は長すぎるため再分割します: {len(chunk) / 1000:.1f}秒"
                        )
                        # チャンクが長すぎる場合は固定長で再分割
                        subchunks = [
                            chunk[j : j + max_chunk_duration]
                            for j in range(0, len(chunk), max_chunk_duration)
                        ]
                        final_chunks.extend(subchunks)
                    else:
                        final_chunks.append(chunk)

                chunks = final_chunks

            logger.info(f"チャンク分割完了: {len(chunks)} チャンク作成")
            return chunks

        except Exception as e:
            logger.error(f"チャンク分割中にエラーが発生: {str(e)}")
            logger.error(traceback.format_exc())
            # エラーが発生した場合は、固定長で強制的に分割
            chunks = [
                audio_data[i : i + max_chunk_duration]
                for i in range(0, len(audio_data), max_chunk_duration)
            ]
            return chunks
