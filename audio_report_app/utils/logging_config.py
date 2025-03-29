from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional


class LoggingConfig:
    """
    ロギング設定の管理クラス

    アプリケーション全体でのロギング設定を提供します。
    """

    # ロガーのキャッシュ
    _loggers: Dict[str, logging.Logger] = {}

    # デフォルトのログレベル
    DEFAULT_LOG_LEVEL = logging.INFO

    # デフォルトのログフォーマット
    DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # デフォルトのログファイル名
    DEFAULT_LOG_FILENAME = "log"

    @classmethod
    def setup_basic_logging(cls, log_level: Optional[int] = None) -> None:
        """
        基本的なロギング設定を行う

        Args:
            log_level: ロギングレベル（指定なしの場合はデフォルト値を使用）
        """
        if log_level is None:
            log_level = cls.DEFAULT_LOG_LEVEL

        # ルートロガーの設定
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # 既存のハンドラをクリア
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # コンソールハンドラの追加
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(cls.DEFAULT_LOG_FORMAT))
        root_logger.addHandler(console_handler)

        # ログディレクトリの作成
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        try:
            # ファイルハンドラの追加（ローテーション付き）
            log_file_path = os.path.join(log_dir, cls.DEFAULT_LOG_FILENAME)
            file_handler = RotatingFileHandler(
                log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
            )
            file_handler.setFormatter(logging.Formatter(cls.DEFAULT_LOG_FORMAT))
            root_logger.addHandler(file_handler)
        except Exception as e:
            # ファイルロギングの設定に失敗した場合はコンソールに出力
            print(f"ログファイルの設定に失敗しました: {str(e)}")

        # matplotlibやPILなどの過剰なログを抑制
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)

        # Streamlitの警告メッセージを抑制
        logging.getLogger("streamlit").setLevel(logging.ERROR)

        # 初期ログ
        root_logger.info("ロギングシステムを初期化しました")

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        指定した名前のロガーを取得する

        Args:
            name: ロガー名

        Returns:
            logging.Logger: 設定済みのロガーインスタンス
        """
        # キャッシュにあればそれを返す
        if name in cls._loggers:
            return cls._loggers[name]

        # 新しいロガーを作成
        logger = logging.getLogger(name)

        # ロガーをキャッシュに保存
        cls._loggers[name] = logger

        return logger

    @staticmethod
    def log_important(message: str) -> None:
        """
        重要なメッセージをログに出力する

        WARNINGレベルでログ出力し、コンソールにも表示します。

        Args:
            message: 出力するメッセージ
        """
        logger = logging.getLogger("IMPORTANT")
        logger.warning(f"[IMPORTANT] {message}")
        print(f"[IMPORTANT] {message}")
