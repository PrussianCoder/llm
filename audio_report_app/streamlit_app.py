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
    # シンプルなテストアプリを表示
    st.title("🎙️ テストモード: 会議記録レポート生成ツール")
    st.write("これはテストモードです。基本機能のみが有効です。")

    # ファイルアップロード
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
        os.unlink(tmp_path)
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
