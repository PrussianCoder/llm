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


# パスと環境変数の設定
def setup_environment():
    """パスと環境変数の初期設定を行う関数"""
    # プロジェクトルートパスの設定
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    logger.info(f"プロジェクトルートパス: {project_root}")

    # アプリパスをPythonパスに追加
    app_path = os.path.join(project_root, "audio_report_app")
    sys.path.append(app_path)
    logger.info(f"アプリパス: {app_path}")

    # Whisperモデルキャッシュのパスを設定
    model_cache_dir = os.path.join(project_root, "models", "whisper")
    os.environ["WHISPER_MODEL_CACHE"] = model_cache_dir
    logger.info(f"Whisperモデルキャッシュディレクトリを設定: {model_cache_dir}")

    # モデルキャッシュディレクトリが存在しない場合は作成
    if not os.path.exists(model_cache_dir):
        os.makedirs(model_cache_dir, exist_ok=True)
        logger.info(f"Whisperモデルディレクトリを作成しました: {model_cache_dir}")

    # HuggingFaceの環境変数を設定
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    logger.info("HuggingFace環境変数を設定しました")

    return model_cache_dir


# 利用可能なパッケージの一覧表示
def log_available_packages():
    """利用可能なパッケージとバージョンをログに表示"""
    try:
        import pkg_resources

        logger.info("利用可能なパッケージ:")
        for pkg in pkg_resources.working_set:
            logger.info(f"  {pkg.key} {pkg.version}")
    except Exception as e:
        logger.error(f"パッケージ一覧取得エラー: {str(e)}")


# テストモード機能
def run_test_mode(model_cache_dir):
    """テストモードでアプリを実行する"""
    import streamlit as st

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
            os.makedirs(whisper_model_path, exist_ok=True)
            st.info(f"指定したモデルディレクトリを作成しました: {whisper_model_path}")

    # モデルファイル一覧を表示
    display_model_files(whisper_model_path)

    # ファイルアップロード
    st.header("音声/動画ファイルテスト")
    uploaded_file = st.file_uploader(
        "音声/動画ファイルをアップロード (.mp3, .mp4, .wav)", type=["mp3", "mp4", "wav"]
    )

    if uploaded_file:
        handle_uploaded_file(uploaded_file)


def display_model_files(model_dir):
    """モデルディレクトリ内のファイルを表示"""
    import streamlit as st

    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth") or f.endswith(".bin")]
        if model_files:
            st.info(f"モデルディレクトリには以下のモデルファイルが見つかりました:")
            for model_file in model_files:
                st.code(f"{model_file}")
        else:
            st.warning(
                f"指定したディレクトリ内にモデルファイルが見つかりません。初回実行時は自動的にダウンロードされます。"
            )
    else:
        st.warning(f"指定したモデルディレクトリは存在しません: {model_dir}")


def handle_uploaded_file(uploaded_file):
    """アップロードされたファイルを処理"""
    import tempfile

    import streamlit as st

    # ファイル情報の表示
    st.info(f"ファイル名: {uploaded_file.name}, サイズ: {uploaded_file.size} bytes")

    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.info(f"一時ファイルに保存しました: {tmp_path}")

    # FFmpegの確認
    if st.button("FFmpegのチェック"):
        check_ffmpeg()

    # Whisperの確認
    if st.button("Whisperモジュールのチェック"):
        check_whisper()

    # Faster Whisperの確認
    if st.button("Faster Whisperモジュールのチェック"):
        check_faster_whisper()

    # 音声の読み込みテスト
    if st.button("音声読み込みテスト"):
        test_audio_loading(tmp_path)

    # 一時ファイルを削除
    try:
        os.unlink(tmp_path)
    except Exception as e:
        st.warning(f"一時ファイルの削除に失敗: {str(e)}")


def check_ffmpeg():
    """FFmpegの利用可能性をチェック"""
    import subprocess

    import streamlit as st

    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        st.code(result.stdout)
        st.success("FFmpegが利用可能です！")
    except Exception as e:
        st.error(f"FFmpegエラー: {str(e)}")
        st.code(traceback.format_exc())


def check_whisper():
    """Whisperモジュールの利用可能性をチェック"""
    import sys

    import streamlit as st

    modules = sys.modules.keys()
    whisper_available = "whisper" in modules

    try:
        import whisper

        st.success(
            f"Whisperモジュールをインポートできました！バージョン: {getattr(whisper, '__version__', '不明')}"
        )

        # モデル一覧
        st.write("利用可能なモデル:")
        st.code(str(whisper.available_models()))

        # モデルディレクトリのチェック
        st.write(f"モデルディレクトリ: {os.environ.get('WHISPER_MODEL_CACHE', '未設定')}")
        if hasattr(whisper, "_download_root"):
            st.write(f"Whisperダウンロードルート: {whisper._download_root}")
        else:
            st.write("Whisperのモデルダウンロードパスが取得できません")
    except ImportError as e:
        st.error(f"Whisperモジュールのインポートに失敗: {str(e)}")

    st.write(f"システムモジュール内に'whisper'がすでに存在: {whisper_available}")

    # インストールされているパッケージを確認
    show_installed_packages("whisper")


def check_faster_whisper():
    """Faster Whisperモジュールの利用可能性をチェック"""
    import streamlit as st

    try:
        from faster_whisper import WhisperModel

        st.success("Faster Whisperモジュールをインポートできました！")

        # モデル情報表示
        st.write("Faster Whisperモデル情報:")
        st.code("利用可能なモデルサイズ: tiny, base, small, medium, large-v1, large-v2, large-v3")

        # モデルディレクトリのチェック
        st.write(f"モデルディレクトリ: {os.environ.get('WHISPER_MODEL_CACHE', '未設定')}")
    except ImportError as e:
        st.error(f"Faster Whisperモジュールのインポートに失敗: {str(e)}")

    # インストールされているパッケージを確認
    show_installed_packages("faster-whisper")


def show_installed_packages(filter_keyword):
    """指定されたキーワードでフィルタリングしたパッケージ一覧を表示"""
    import pkg_resources
    import streamlit as st

    filtered_packages = [
        p for p in pkg_resources.working_set if filter_keyword.lower() in p.key.lower()
    ]

    st.write(f"{filter_keyword}関連のインストール済みパッケージ:")
    for pkg in filtered_packages:
        st.code(f"{pkg.key} {pkg.version}")


def test_audio_loading(file_path):
    """音声ファイルの読み込みをテスト"""
    import streamlit as st

    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(file_path)
        st.success(
            f"音声を読み込みました: 長さ {len(audio)/1000:.2f}秒, チャンネル数: {audio.channels}"
        )
    except Exception as e:
        st.error(f"音声読み込みエラー: {str(e)}")
        st.code(traceback.format_exc())


def run_main_app():
    """メインアプリケーションを実行"""
    import streamlit as st

    # メインアプリケーションのインポートとセットアップ
    from services.audio_processor import AudioProcessorV2

    from main import AudioReportApp

    logger.info("アプリケーションクラスをインポートしました")

    # アプリケーションの実行
    app = AudioReportApp(AudioProcessorV2())
    logger.info("アプリケーションインスタンスを作成しました")

    app.run()
    logger.info("アプリケーションの実行が完了しました")


# メイン処理
def main():
    try:
        # 環境設定
        model_cache_dir = setup_environment()

        # Streamlitをインポート
        import streamlit as st

        st.set_page_config(page_title="会議記録レポート生成ツール", page_icon="🎙️", layout="wide")
        logger.info("Streamlitページ設定を完了しました")

        # パッケージ情報のログ出力
        log_available_packages()

        # テストモード
        TEST_MODE = True

        if TEST_MODE:
            run_test_mode(model_cache_dir)
        else:
            run_main_app()

    except Exception as e:
        logger.error(f"アプリケーション実行エラー: {str(e)}")
        logger.error(traceback.format_exc())

        # Streamlitがすでにインポートされている場合はエラーを表示
        try:
            import streamlit as st

            st.error(f"アプリケーション起動エラー: {str(e)}")
            st.error("詳細なエラー情報は以下の通りです:")
            st.code(traceback.format_exc())
        except:
            pass  # Streamlitが使えない場合は標準エラー出力のみ


if __name__ == "__main__":
    main()
