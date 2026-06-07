# rust-llm

`rust-llm` は、RustとCandleを利用して実装された、シンプルな文字単位（Character-level）のTransformer言語モデルライブラリです。

## 外部クレートとしての利用方法

他のRustプロジェクトからこのライブラリを利用するには、利用側プロジェクトの `Cargo.toml` の `[dependencies]` に追加します。
ローカル環境にある場合はパスを指定してインポートできます。

```toml
[dependencies]
rust-llm = { path = "../path/to/rust-llm" } # 実際の配置パスに合わせて変更してください
```

GitHubなどのリモートリポジトリで管理している場合は、git URLを使って指定することも可能です。

```toml
[dependencies]
rust-llm = { git = "https://github.com/your-username/rust-llm.git" }
```

## 主要な関数

本ライブラリは、モデルとの対話（推論）や小規模な学習（ファインチューニング）を行うために以下の主要な関数を提供しています。

### `chat` 関数

ユーザーからのテキスト入力を受け取り、AIとしてテキスト応答（推論結果）を生成して返します。

```rust
pub fn chat(
    transformer: &Transformer,
    device: &Device,
    tokenizer: &mut CharTokenizer,
    text: &str,
) -> Result<String, Box<dyn std::error::Error>>
```

* **`transformer`**: 推論に使用するTransformerモデルのインスタンス。
* **`device`**: テンソル演算を行うデバイス（CPUまたはGPU）。
* **`tokenizer`**: 文字とトークンIDを相互変換するトークナイザー。
* **`text`**: ユーザーからの入力文字列。

### `meaning` 関数

特定の単語（word）とその意味（meaning_text）のペアをモデルに学習させます。内部的には「{word}の意味は{meaning_text}です。」という文脈を作成し、重みの更新を行います。

```rust
pub fn meaning(
    transformer: &Transformer,
    varmap: &VarMap,
    device: &Device,
    tokenizer: &mut CharTokenizer,
    word: &str,
    meaning_text: &str,
) -> Result<(), Box<dyn std::error::Error>>
```

* **`varmap`**: モデルの学習可能な変数を管理するマップ（Optimizerでの重み更新に使用）。
* **`word`**: 学習させたい対象の単語。
* **`meaning_text`**: 対象の単語の意味や説明のテキスト。

### `training` 関数

任意の入力（input）と出力（output）のペアをモデルに学習させます。基本的な仕組みは `meaning` 関数と同様ですが、より汎用的なテキストペアの学習に使用できます。

```rust
pub fn training(
    transformer: &Transformer,
    varmap: &VarMap,
    device: &Device,
    tokenizer: &mut CharTokenizer,
    input: &str,
    output: &str,
) -> Result<(), Box<dyn std::error::Error>>
```

* **`input`**: 学習のトリガーとなる入力テキスト。
* **`output`**: 入力に対してモデルに期待する出力のテキスト。

## 実装例

外部クレートから呼び出して使用する場合の簡単なイメージです。

```rust
use rust_llm::{chat, meaning, training, CharTokenizer};

// ※ 実行前に Transformer, Device, VarMap などの初期化を行ってください。

// 1. 言葉の意味を学習させる
// meaning(&transformer, &varmap, &device, &mut tokenizer, "Rust", "安全で高速なシステムプログラミング言語")?;

// 2. 学習結果を踏まえてチャットで推論する
// let response = chat(&transformer, &device, &mut tokenizer, "Rust")?;
```
