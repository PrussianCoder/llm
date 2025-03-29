from __future__ import annotations

import os
import tempfile
from typing import Any, BinaryIO

import streamlit as st
from utils.logging_config import LoggingConfig

# ロガーの取得
logger = LoggingConfig.get_logger("FileHandler")


class FileHandler:
    """
    ファイル操作を行うユーティリティクラス

    一時ファイルの作成や削除などの機能を提供します。
    """

    @staticmethod
    def save_uploaded_file(uploaded_file) -> str:
        """
        アップロードされたファイルを一時ファイルとして保存する

        Args:
            uploaded_file: アップロードされたファイルオブジェクト

        Returns:
            str: 一時ファイルのパス
        """
        # 一時ファイルを作成
        suffix = "." + uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            # ファイルの内容を一時ファイルに書き込む
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        return temp_path

    @staticmethod
    def create_temp_file(uploaded_file: Any) -> str:
        """
        アップロードされたファイルを一時ファイルとして保存する

        Args:
            uploaded_file: Streamlitのアップロードファイルオブジェクト

        Returns:
            str: 一時ファイルのパス
        """
        # ファイル拡張子を取得
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            # ファイルの内容を一時ファイルに書き込む
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        logger.info(f"一時ファイルを作成しました: {temp_path}")
        return temp_path

    @staticmethod
    def delete_file(file_path: str) -> None:
        """
        ファイルを削除する

        Args:
            file_path: 削除するファイルのパス
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"ファイルを削除しました: {file_path}")
            else:
                logger.warning(f"削除対象のファイルが存在しません: {file_path}")
        except Exception as e:
            logger.error(f"ファイル削除中にエラーが発生しました: {str(e)}")

    @staticmethod
    def get_file_extension(file_name: str) -> str:
        """
        ファイル名から拡張子を取得する

        Args:
            file_name: ファイル名

        Returns:
            str: 拡張子（ドットを含む）
        """
        return os.path.splitext(file_name)[1]

    def generate_download_link(
        self, content: str, file_name: str, mime_type: str = "text/plain"
    ) -> str:
        """
        ダウンロード用のリンクを生成する

        Args:
            content: ダウンロードするコンテンツ
            file_name: ダウンロードファイル名
            mime_type: MIMEタイプ

        Returns:
            str: ダウンロード用のHTML link
        """
        b64_content = self._to_base64(content, mime_type)
        download_link = f'<a href="data:{mime_type};base64,{b64_content}" download="{file_name}">ダウンロード: {file_name}</a>'
        return download_link

    def _to_base64(self, content: str, mime_type: str) -> str:
        """
        コンテンツをBase64エンコードする

        Args:
            content: エンコードするコンテンツ
            mime_type: MIMEタイプ

        Returns:
            str: Base64エンコードされた文字列
        """
        import base64

        try:
            # テキストをUTF-8でエンコードしてBase64変換
            encoded_content = base64.b64encode(content.encode("utf-8")).decode()
            return encoded_content
        except Exception as e:
            logger.error(f"Base64エンコード中にエラーが発生しました: {str(e)}")
            return ""
