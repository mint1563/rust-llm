# rust-llm

RustとHugging Faceの機械学習フレームワークである[Candle](https://github.com/huggingface/candle)を使用した、GPTのようなTransformerデコーダーの実装です。

## 概要

このプロジェクトは、Rustを使ってゼロから大規模言語モデル（LLM）のアーキテクチャを構築する方法を学ぶためのサンプルです。因果的自己注意機構（Causal Self-Attention）、フィードフォワードネットワーク、およびテキスト生成ループを備えた基本的なTransformerデコーダーモデルを実装しています。

また、本プロジェクトではプロンプトの指示に従い、チャット機能、意味の学習、入出力の学習、思考のログ保存機能を備えています。

## 機能

- **Transformerデコーダー:**
  - Causal Self-AttentionとMLPを用いた基本的なTransformer実装。
  - candle-core と candle-nn を利用した効率的なテンソル演算。
- **学習データの保存:** 実行時にモデルの重みを model_weights.safetensors として保存します。
- **ベクトルデータの可視化:** 単語の埋め込み（Embedding）ベクトルを embeddings.csv として出力し、可視化に利用できるようにしています。
- **chat関数:** テキストを入力するとTransformerで学習された言語モデルが出力する関数です。
- **meaning関数:** 単語の意味を学習する機能を提供します。
- **training関数:** 入力と出力のペアを学習する機能を提供します。
- **thinkingログ機能:** 実行中の思考内容（処理内容）を  hinking.log に保存し、次回起動時に参照してログをコンソールに出力します。

## 使い方

以下のコマンドを実行してプログラムを開始します。

'''bash
cargo run
'''

実行すると、モデルの重みとベクトルデータが保存され、続いて meaning関数、 raining関数、chat関数のデモンストレーションが順に行われます。また、各プロセスの実行ログが  hinking.log に追記されていきます。

## プロジェクト構成

- src/model.rs: Transformerモデルのコア実装が含まれています。
- src/main.rs: chat、meaning、 raining の各関数と、ログ機構、メインの実行ループが含まれています。
- hinking.log: プログラムの思考・処理ログが保存されるファイルです。
