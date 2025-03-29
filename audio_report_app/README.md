# Audio Report App

音声/動画ファイルから文字起こしを行い、AIによるレポート生成と質問応答ができるStreamlitアプリケーション。

## 機能

- 様々な音声/動画フォーマットのファイルに対応
- 複数の音声認識エンジンをサポート
  - Whisper
  - Faster Whisper
  - Google Speech Recognition
  - CMU Sphinx
- 音声前処理機能
  - ノイズ除去
  - 無音区間除去
  - 音声強化処理
- 長時間音声の分割処理と並列認識
- リアルタイム認識結果表示
- OpenAI APIを利用したレポート生成
  - 要約
  - 議事録
  - トピック抽出
- 音声内容に関する質問応答機能

## インストール

必要なパッケージをインストールします：

```bash
pip install -r requirements.txt
```

FFmpegが必要です：

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu
sudo apt-get install ffmpeg

# Windows
# FFmpegをダウンロードして環境変数に追加
```

## 使い方

アプリケーションを起動するには：

```bash
python -m audio_report_app.main
```

または：

```bash
streamlit run audio_report_app/main.py
```

## 設定

1. 左サイドバーから各種設定を変更できます
   - 音声認識エンジン選択
   - 言語設定
   - 音声処理オプション
   - レポートタイプ

2. OpenAI APIキーの設定
   - 左上の「APIキー設定」から入力
   - または環境変数 `OPENAI_API_KEY` を設定

## プロジェクト構成

```
audio_report_app/
├── interfaces/      # インターフェース定義
├── services/        # サービス実装（音声処理、テキスト処理等）
├── ui/              # ユーザーインターフェース
├── utils/           # ユーティリティ関数
├── settings/        # 設定ファイル
├── data/            # データ保存用
├── logs/            # ログファイル
└── main.py          # アプリケーションエントリーポイント
```

## 対応言語

- 英語 (en)
- 日本語 (ja)
- その他複数言語 (Whisperエンジン使用時)

## 注意事項

- 音声認識精度は録音品質やエンジンによって異なります
- Whisperモデルは初回使用時にダウンロードされます
- 長時間の音声処理はリソースを多く消費します
- 音声認識結果に繰り返しパターンが発生する場合があります（自動修正機能あり）

## ライセンス

MIT 