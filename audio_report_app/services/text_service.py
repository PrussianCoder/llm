from __future__ import annotations

import logging
import os
from typing import Dict, Optional

from interfaces.i_text_processor import ITextProcessor
from openai import OpenAI
from utils.logging_config import LoggingConfig

# ロガーの取得
logger = LoggingConfig.get_logger("TextService")


class TextService(ITextProcessor):
    """
    テキストの処理とレポート生成を行う実装クラス

    OpenAI APIを使用してレポートやチャット応答を生成します。
    """

    def __init__(self) -> None:
        """TextServiceのインスタンスを初期化"""
        # レポートタイプとプロンプトのマッピング
        self.report_prompts: Dict[str, str] = {
            "会議録": (
                "以下の会議の文字起こしに基づいて、会議録を作成してください。参加者、日時、主要な議題、議論内容、決定事項を含めてください。"
            ),
            "議事録": (
                "以下の会議の文字起こしに基づいて、プロフェッショナルな議事録を作成してください。参加者（話者が特定できる場合）、議論されたトピック、決定事項、アクションアイテム、次のステップを含めてください。"
            ),
            "要約": (
                "以下の会議の文字起こしに基づいて、最大500単語の要約レポートを作成してください。重要なポイント、議論された主要なトピック、決定事項に焦点を当ててください。"
            ),
            "アクションアイテム": (
                "以下の会議の文字起こしを分析し、すべてのアクションアイテム、タスク、コミットメント、締め切り、担当者を箇条書きリストで抽出してください。次のフォーマットを使用してください：[アクション] - [担当者（わかる場合）] - [締め切り（わかる場合）]"
            ),
            "Q&A抽出": (
                "以下の会議の文字起こしから、すべての質問と回答のペアを抽出し、Q&A形式でまとめてください。関連する質問と回答をグループ化し、トピックごとに整理してください。"
            ),
        }

    def generate_report(
        self,
        transcript: str,
        model: str = "gpt-3.5-turbo",
        report_type: str = "要約",
        openai_api_key: Optional[str] = None,
    ) -> str:
        """
        テキスト文字起こしからレポートを生成する

        Args:
            transcript: 文字起こしテキスト
            model: OpenAIモデル名
            report_type: 生成するレポートタイプ
            openai_api_key: OpenAI APIキー

        Returns:
            str: 生成されたレポート
        """
        if not transcript or len(transcript.split()) < 10:
            logger.warning("文字起こしテキストが短すぎます。有効なレポートを生成できません。")
            return "文字起こしテキストが不十分なため、レポートを生成できませんでした。より長い音声データを提供するか、認識設定を調整してください。"

        if not openai_api_key:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error("OpenAI APIキーが設定されていません")
                return "APIキーが設定されていないため、レポートを生成できませんでした。"

        # プロンプトを取得
        prompt = self.report_prompts.get(report_type, self.report_prompts["要約"])

        # APIリクエストを実行
        try:
            logger.info(f"OpenAIに{report_type}の生成をリクエスト中...")

            # OpenAIクライアントの初期化
            client = OpenAI(api_key=openai_api_key)

            # API呼び出し
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"あなたは会議の内容を分析して{report_type}を作成する専門家です。"
                        ),
                    },
                    {"role": "user", "content": f"{prompt}\n\n文字起こし：\n{transcript}"},
                ],
                temperature=0.5,
                max_tokens=1500,
            )

            # レスポンスからテキストを抽出
            report = response.choices[0].message.content

            logger.info(f"{report_type}の生成が完了しました")
            return report

        except Exception as e:
            logger.error(f"レポート生成中にエラーが発生しました: {str(e)}", exc_info=True)

            # フォールバック: gpt-3.5-turboで再試行
            if model != "gpt-3.5-turbo":
                logger.info(f"{model}でエラーが発生したため、gpt-3.5-turboで再試行します")
                try:
                    return self.generate_report(
                        transcript, "gpt-3.5-turbo", report_type, openai_api_key
                    )
                except Exception as fallback_error:
                    logger.error(f"フォールバックでもエラーが発生しました: {str(fallback_error)}")

            return f"レポート生成中にエラーが発生しました: {str(e)}"

    def generate_chat_response(
        self,
        question: str,
        transcript: str,
        report: str,
        model: str = "gpt-3.5-turbo",
        openai_api_key: Optional[str] = None,
    ) -> str:
        """
        チャット質問に対する回答を生成する

        Args:
            question: ユーザーからの質問
            transcript: 文字起こしテキスト
            report: 生成されたレポート
            model: OpenAIモデル名
            openai_api_key: OpenAI APIキー

        Returns:
            str: 生成された回答
        """
        if not openai_api_key:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error("OpenAI APIキーが設定されていません")
                return "APIキーが設定されていないため、回答を生成できませんでした。"

        try:
            logger.info("チャット応答の生成をリクエスト中...")

            # システムプロンプトの準備
            system_prompt = f"""
            以下は音声データの文字起こしとそのレポートです。これらの情報に基づいて質問に回答してください。
            
            ## 文字起こし:
            {transcript}
            
            ## レポート:
            {report}
            """

            # OpenAIクライアントの初期化
            client = OpenAI(api_key=openai_api_key)

            # API呼び出し
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            # レスポンスからテキストを抽出
            answer = response.choices[0].message.content

            logger.info("チャット応答の生成が完了しました")
            return answer

        except Exception as e:
            logger.error(f"チャット応答生成中にエラーが発生しました: {str(e)}", exc_info=True)
            return f"回答の生成中にエラーが発生しました: {str(e)}"

    @classmethod
    def create(cls) -> ITextProcessor:
        """
        TextServiceのインスタンスを作成する

        Returns:
            ITextProcessor: 新しいTextServiceインスタンス
        """
        return TextService()
