#!/usr/bin/env python3
"""
Streamlitアプリケーションのランチャー
"""
import logging
import os
import sys
import traceback

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("streamlit_app")

# ルートパスを追加
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    logger.info(f"プロジェクトルートパス: {project_root}")
except Exception as e:
    logger.error(f"パス設定エラー: {str(e)}")
    traceback.print_exc()

# Whisperモデルキャッシュのパスを設定
model_cache_dir = os.path.join(project_root, "models", "whisper")
os.environ["WHISPER_MODEL_CACHE"] = model_cache_dir
logger.info(f"Whisperモデルキャッシュディレクトリを設定: {model_cache_dir}")

# モデルキャッシュディレクトリが存在しない場合は作成
try:
    if not os.path.exists(model_cache_dir):
        os.makedirs(model_cache_dir, exist_ok=True)
        logger.info(f"Whisperモデルディレクトリを作成しました: {model_cache_dir}")
except Exception as e:
    logger.error(f"モデルディレクトリ作成エラー: {str(e)}")
    traceback.print_exc()

# まず最初にStreamlitをインポート
try:
    import streamlit as st

    logger.info("Streamlitをインポートしました")
except Exception as e:
    logger.error(f"Streamlitインポートエラー: {str(e)}")
    traceback.print_exc()

# ページ設定を最初のStreamlit命令として設定
try:
    st.set_page_config(page_title="会議記録レポート生成ツール", page_icon="🎙️", layout="wide")
    logger.info("Streamlitページ設定を完了しました")
except Exception as e:
    logger.error(f"ページ設定エラー: {str(e)}")
    traceback.print_exc()

# audio_report_appをPythonパスに追加
try:
    app_path = os.path.join(project_root, "audio_report_app")
    sys.path.append(app_path)
    logger.info(f"アプリパス: {app_path}")

    # 利用可能なパッケージとバージョンを表示
    import pkg_resources

    logger.info("利用可能なパッケージ:")
    for pkg in pkg_resources.working_set:
        logger.info(f"  {pkg.key} {pkg.version}")
except Exception as e:
    logger.error(f"パッケージ一覧取得エラー: {str(e)}")
    traceback.print_exc()

# 簡易テストモード
TEST_MODE = False

if TEST_MODE:
    try:
        # シンプルなテストアプリを表示
        st.title("🎙️ テストモード: 会議記録レポート生成ツール")
        st.write("これはテストモードです。基本機能のみが有効です。")

        # Whisper設定セクション
        st.header("Whisperモデル設定")
        whisper_model_path = st.text_input(
            "Whisperモデルディレクトリのパス",
            value=model_cache_dir,
            help="Whisperモデルを保存するディレクトリのパスを指定します。既に.pthファイルがある場合はそれを使用します。",
        )

        if st.button("モデルパスを更新"):
            os.environ["WHISPER_MODEL_CACHE"] = whisper_model_path
            st.success(f"Whisperモデルパスを更新しました: {whisper_model_path}")

            # モデルディレクトリが存在するか確認
            if not os.path.exists(whisper_model_path):
                try:
                    os.makedirs(whisper_model_path, exist_ok=True)
                    st.info(f"指定したモデルディレクトリを作成しました: {whisper_model_path}")
                except Exception as e:
                    st.error(f"モデルディレクトリの作成に失敗: {str(e)}")

        # モデルファイル一覧を表示
        if os.path.exists(whisper_model_path):
            model_files = [
                f
                for f in os.listdir(whisper_model_path)
                if f.endswith(".pth") or f.endswith(".bin")
            ]
            if model_files:
                st.info(f"モデルディレクトリには以下のモデルファイルが見つかりました:")
                for model_file in model_files:
                    st.code(f"{model_file}")
            else:
                st.warning(
                    f"指定したディレクトリ内にモデルファイルが見つかりません。初回実行時は自動的にダウンロードされます。"
                )
        else:
            st.warning(f"指定したモデルディレクトリは存在しません: {whisper_model_path}")

        # ファイルアップロード
        st.header("音声/動画ファイルテスト")
        uploaded_file = st.file_uploader(
            "音声/動画ファイルをアップロード (.mp3, .mp4, .wav)", type=["mp3", "mp4", "wav"]
        )

        if uploaded_file:
            # ファイル情報の表示
            st.info(f"ファイル名: {uploaded_file.name}, サイズ: {uploaded_file.size} bytes")

            # 一時ファイルに保存
            import tempfile

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
            ) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            st.info(f"一時ファイルに保存しました: {tmp_path}")

            # FFmpegの確認
            if st.button("FFmpegのチェック"):
                try:
                    import subprocess

                    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
                    st.code(result.stdout)
                    st.success("FFmpegが利用可能です！")
                except Exception as e:
                    st.error(f"FFmpegエラー: {str(e)}")
                    st.code(traceback.format_exc())

            # Whisperの確認
            if st.button("Whisperモジュールのチェック"):
                try:
                    import sys

                    modules = sys.modules.keys()
                    whisper_available = "whisper" in modules

                    try:
                        import whisper

                        whisper_import_success = True
                        st.success(
                            f"Whisperモジュールをインポートできました！バージョン: {getattr(whisper, '__version__', '不明')}"
                        )

                        # モデル一覧
                        st.write("利用可能なモデル:")
                        st.code(str(whisper.available_models()))

                        # モデルディレクトリのチェック
                        st.write(
                            f"モデルディレクトリ: {os.environ.get('WHISPER_MODEL_CACHE', '未設定')}"
                        )
                        if hasattr(whisper, "_download_root"):
                            st.write(f"Whisperダウンロードルート: {whisper._download_root}")
                        else:
                            st.write("Whisperのモデルダウンロードパスが取得できません")
                    except ImportError as e:
                        whisper_import_success = False
                        st.error(f"Whisperモジュールのインポートに失敗: {str(e)}")

                    st.write(f"システムモジュール内に'whisper'がすでに存在: {whisper_available}")
                    st.write(f"Whisperのインポート成功: {whisper_import_success}")

                    # インストールされているパッケージを確認
                    import pkg_resources

                    whisper_packages = [
                        p for p in pkg_resources.working_set if "whisper" in p.key.lower()
                    ]
                    st.write("Whisper関連のインストール済みパッケージ:")
                    for pkg in whisper_packages:
                        st.code(f"{pkg.key} {pkg.version}")

                except Exception as e:
                    st.error(f"Whisperチェックエラー: {str(e)}")
                    st.code(traceback.format_exc())

            # Faster Whisperの確認
            if st.button("Faster Whisperモジュールのチェック"):
                try:
                    try:
                        from faster_whisper import WhisperModel

                        st.success("Faster Whisperモジュールをインポートできました！")

                        # モデル情報表示
                        st.write("Faster Whisperモデル情報:")
                        st.code(
                            "利用可能なモデルサイズ: tiny, base, small, medium, large-v1, large-v2, large-v3"
                        )

                        # モデルディレクトリのチェック
                        st.write(
                            f"モデルディレクトリ: {os.environ.get('WHISPER_MODEL_CACHE', '未設定')}"
                        )
                    except ImportError as e:
                        st.error(f"Faster Whisperモジュールのインポートに失敗: {str(e)}")

                    # インストールされているパッケージを確認
                    import pkg_resources

                    faster_whisper_packages = [
                        p
                        for p in pkg_resources.working_set
                        if "faster" in p.key.lower() and "whisper" in p.key.lower()
                    ]
                    st.write("Faster Whisper関連のインストール済みパッケージ:")
                    for pkg in faster_whisper_packages:
                        st.code(f"{pkg.key} {pkg.version}")

                except Exception as e:
                    st.error(f"Faster Whisperチェックエラー: {str(e)}")
                    st.code(traceback.format_exc())

            # 音声の読み込みテスト
            if st.button("音声読み込みテスト"):
                try:
                    from pydub import AudioSegment

                    audio = AudioSegment.from_file(tmp_path)
                    st.success(
                        f"音声を読み込みました: 長さ {len(audio)/1000:.2f}秒, チャンネル数: {audio.channels}"
                    )
                except Exception as e:
                    st.error(f"音声読み込みエラー: {str(e)}")
                    st.code(traceback.format_exc())

            # 一時ファイルを削除
            try:
                os.unlink(tmp_path)
            except Exception as e:
                st.warning(f"一時ファイルの削除に失敗: {str(e)}")
    except Exception as e:
        st.error(f"テストモード実行エラー: {str(e)}")
        st.code(traceback.format_exc())
else:
    try:
        # メインアプリケーションのインポートとセットアップ
        from services.audio_processor import AudioProcessorV2

        from main import AudioReportApp

        logger.info("アプリケーションクラスをインポートしました")

        # アプリケーションの実行
        app = AudioReportApp(AudioProcessorV2())
        logger.info("アプリケーションインスタンスを作成しました")

        app.run()
        logger.info("アプリケーションの実行が完了しました")
    except Exception as e:
        st.error(f"アプリケーション実行エラー: {str(e)}")
        st.code(traceback.format_exc())
