#!/usr/bin/env python3
"""
Streamlitアプリケーションのランチャー
"""
import os
import sys

# ルートパスを追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# まず最初にStreamlitをインポート
import streamlit as st

# ページ設定を最初のStreamlit命令として設定
st.set_page_config(page_title="会議記録レポート生成ツール", page_icon="🎙️", layout="wide")

# audio_report_appをPythonパスに追加
app_path = os.path.join(project_root, "audio_report_app")
sys.path.append(app_path)

# メインアプリケーションのインポートとセットアップ
from audio_report_app.main import AudioReportApp
from audio_report_app.services.audio_processor import AudioProcessorV2

# アプリケーションの実行
app = AudioReportApp(AudioProcessorV2())
app.run()
