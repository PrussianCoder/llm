"""
pydubライブラリのutils.pyの代替実装

FFmpegやffprobeが利用できない環境でも基本的な音声処理を可能にするユーティリティ関数
"""

import json
import logging
import os
import subprocess
import wave
from typing import Any, Dict, Optional

logger = logging.getLogger("pydub_utils")


def mediainfo_fallback(filepath: str) -> Dict[str, Any]:
    """
    ffprobeが利用できない場合のフォールバックとしてメディア情報を収集する

    Args:
        filepath: ファイルパス

    Returns:
        Dict[str, Any]: メディア情報の辞書
    """
    ext = os.path.splitext(filepath)[1].lower()

    info = {
        "filepath": filepath,
        "format": {
            "duration": "0",  # 初期値
            "bit_rate": "0",
            "filename": os.path.basename(filepath),
            "format_name": ext.replace(".", ""),
        },
        "streams": [],
    }

    # WAVファイルの場合、waveモジュールから情報を取得
    if ext == ".wav":
        try:
            with wave.open(filepath, "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()

                # durationを計算（秒）
                duration = n_frames / float(frame_rate)
                info["format"]["duration"] = str(duration)

                # ビットレートを推定
                bit_rate = channels * sample_width * 8 * frame_rate
                info["format"]["bit_rate"] = str(bit_rate)

                # ストリーム情報
                stream_info = {
                    "codec_type": "audio",
                    "codec_name": "pcm",
                    "sample_rate": str(frame_rate),
                    "channels": channels,
                    "bits_per_sample": sample_width * 8,
                }
                info["streams"].append(stream_info)
        except Exception as e:
            logger.error(f"WAVファイル情報の取得に失敗: {str(e)}")

    # MP3ファイルなど、他の形式の場合は推定値を設定
    else:
        # ファイルサイズからおおよその長さを推定
        try:
            file_size = os.path.getsize(filepath)
            # 仮の平均ビットレート（128kbps）と推定時間
            bit_rate = 128000
            info["format"]["bit_rate"] = str(bit_rate)

            # 時間の推定（秒）: ファイルサイズ(bytes) / (ビットレート(bits/sec) / 8)
            estimated_duration = file_size / (bit_rate / 8)
            info["format"]["duration"] = str(estimated_duration)

            # 推定ストリーム情報
            stream_info = {
                "codec_type": "audio",
                "codec_name": ext.replace(".", ""),
                "sample_rate": "44100",  # 一般的な値
                "channels": 2,  # ステレオと仮定
                "bits_per_sample": 16,  # 16ビットと仮定
            }
            info["streams"].append(stream_info)
        except Exception as e:
            logger.error(f"ファイルサイズの取得に失敗: {str(e)}")

    logger.warning(f"FFprobe代替機能を使用: {filepath} の基本情報を推定しました")
    return info


def mediainfo_json_safe(filepath: str, read_ahead_limit: Optional[int] = None) -> Dict[str, Any]:
    """
    ffprobeを使用してメディア情報をJSON形式で取得するか、失敗した場合はフォールバック処理を使用

    Args:
        filepath: ファイルパス
        read_ahead_limit: 先読み制限（使用しない）

    Returns:
        Dict[str, Any]: メディア情報の辞書
    """
    from subprocess import PIPE, Popen

    try:
        # ffprobeコマンドの作成
        command = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            filepath,
        ]

        # ffprobeの実行
        res = Popen(command, stdout=PIPE, stderr=PIPE)
        output, stderr = res.communicate()

        # 実行結果のチェック
        if res.returncode != 0:
            logger.warning(f"ffprobeが失敗: {stderr.decode()}")
            return mediainfo_fallback(filepath)

        # JSONの解析
        return json.loads(output.decode("utf-8"))
    except Exception as e:
        logger.warning(f"ffprobe処理中にエラー発生: {str(e)}")
        return mediainfo_fallback(filepath)


# pydubのAudioSegmentクラスのロード時にこの関数を使用するためのセットアップ
def setup_pydub_fallback():
    """
    pydubライブラリのutils.mediainfo_json関数を上書きして、
    FFmpegが利用できない環境でも基本的な機能を使えるようにする

    注意: この関数を呼び出すとpydubの内部実装を変更します
    """
    try:
        import pydub.utils

        # 元の関数を保存
        original_mediainfo_json = pydub.utils.mediainfo_json

        # 元の関数がある場合は維持し、エラー時のみフォールバック
        def patched_mediainfo_json(filepath, read_ahead_limit=None):
            try:
                return original_mediainfo_json(filepath, read_ahead_limit)
            except (FileNotFoundError, subprocess.SubprocessError) as e:
                logger.warning(f"オリジナルのmediainfo_jsonがエラー: {str(e)}")
                return mediainfo_fallback(filepath)

        # pydubの関数を置き換え
        pydub.utils.mediainfo_json = patched_mediainfo_json
        logger.info("pydubのmediainfo_json関数を安全なバージョンに置き換えました")
    except ImportError:
        logger.error("pydubライブラリをインポートできないため、フォールバックの設定に失敗しました")
    except Exception as e:
        logger.error(f"pydubフォールバックの設定中にエラー: {str(e)}")
