from __future__ import annotations

from typing import Any

import streamlit as st
from interfaces.i_ui_component import IReportComponent


class ReportComponent(IReportComponent):
    """
    レポート表示用のUIコンポーネント

    生成されたレポートの表示やダウンロード機能を提供します。
    """

    def __init__(self) -> None:
        """ReportComponentのインスタンスを初期化"""
        pass

    def render(self) -> Any:
        """
        UIコンポーネントをレンダリングする

        Returns:
            Any: レンダリング結果
        """
        st.subheader("生成されたレポート")
        return None

    def show_report(self, report: str, report_type: str, file_name: str) -> None:
        """
        生成されたレポートを表示する

        Args:
            report: レポートのテキスト
            report_type: レポートタイプ
            file_name: 元のファイル名
        """
        st.subheader("生成されたレポート")

        # レポート内容の表示
        st.markdown(report)

        # レポートのダウンロードボタン
        st.download_button(
            label="レポートをダウンロード",
            data=report,
            file_name=f"{report_type}_{file_name.split('.')[0]}.txt",
            mime="text/plain",
        )

    @classmethod
    def create(cls) -> IReportComponent:
        """
        ReportComponentのインスタンスを作成する

        Returns:
            IReportComponent: 新しいReportComponentインスタンス
        """
        return ReportComponent()
