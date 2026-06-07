pub mod model;

use candle_core::{Device, Tensor};
use candle_nn::{Optimizer, VarMap};
use model::Transformer;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

/// 文字単位のトークナイザー（Character-level Tokenizer）
/// テキスト内の文字(char)とモデルが処理できるトークンID(u32)の相互変換を管理します。
pub struct CharTokenizer {
    ch_to_id: HashMap<char, u32>,
    id_to_ch: HashMap<u32, char>,
    next_id: u32,
}

impl CharTokenizer {
    /// 初期状態の空のトークナイザーを作成します。
    pub fn new() -> Self {
        Self {
            ch_to_id: HashMap::new(),
            id_to_ch: HashMap::new(),
            next_id: 0,
        }
    }

    /// 文字列を受け取り、トークンIDのリスト(Vec<u32>)にエンコードします。
    /// 未知の文字が見つかった場合は、辞書に新しいIDを自動登録します（最大9999まで）。
    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| {
                if let Some(&id) = self.ch_to_id.get(&c) {
                    id
                } else {
                    let id = self.next_id;
                    self.ch_to_id.insert(c, id);
                    self.id_to_ch.insert(id, c);
                    if self.next_id < 9999 {
                        self.next_id += 1;
                    }
                    id
                }
            })
            .collect()
    }

    /// トークンIDのリストを元の文字列に復元(デコード)します。
    /// 辞書に存在しない未知のIDは '?' に置き換えられます。
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .map(|&t| self.id_to_ch.get(&t).copied().unwrap_or('?'))
            .collect()
    }

    /// トークナイザーの辞書(文字とIDの対応)をCSV形式でファイルに保存します。
    pub fn save(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(filename)?;
        for (ch, id) in &self.ch_to_id {
            writeln!(file, "{},{}", *ch as u32, id)?;
        }
        Ok(())
    }

    /// ファイルからトークナイザーの辞書を読み込み、状態を復元します。
    pub fn load(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut t = Self::new();
        if Path::new(filename).exists() {
            let file = File::open(filename)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() == 2 {
                    if let (Ok(c_u32), Ok(id)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>())
                    {
                        if let Some(ch) = char::from_u32(c_u32) {
                            t.ch_to_id.insert(ch, id);
                            t.id_to_ch.insert(id, ch);
                            if id >= t.next_id {
                                t.next_id = id + 1;
                            }
                        }
                    }
                }
            }
        }
        Ok(t)
    }
}

/// LLMの「思考プロセス」などを記録するためのログファイルにメッセージを追記します。
pub fn append_thinking_log(log_msg: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("thinking.log")?;
    writeln!(file, "{}", log_msg)?;
    Ok(())
}

/// 記録されている「思考プロセス」のログ全体を読み込みます。
pub fn read_thinking_log() -> Result<String, Box<dyn std::error::Error>> {
    if Path::new("thinking.log").exists() {
        let mut file = File::open("thinking.log")?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        Ok(contents)
    } else {
        Ok(String::new())
    }
}

/// モデルを用いて、与えられた入力から続くテキスト（トークン）を生成します。
/// max_new_tokens で指定した最大回数だけループし、1トークンずつ予測を追加します。
pub fn generate_text(
    transformer: &Transformer,
    device: &Device,
    input_tokens: Vec<u32>,
    max_new_tokens: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut current_tokens = input_tokens.clone();

    for _ in 0..max_new_tokens {
        let seq_len = current_tokens.len();
        let input = Tensor::from_vec(current_tokens.clone(), (1, seq_len), device)?;

        // モデルに順伝播させて予測スコア(Logits)を取得
        let logits = transformer.forward(&input)?;
        // シーケンスの最後のトークンの予測結果を取得
        let last_token_logits = logits.squeeze(0)?.get(seq_len - 1)?;
        // 最も確率の高いトークンのIDを選択（貪欲法/Argmax）
        let next_token = last_token_logits.argmax(0)?.to_scalar::<u32>()?;

        current_tokens.push(next_token);
    }

    Ok(current_tokens)
}

/// モデル内の埋め込み（Embedding）テンソルの中身を取り出し、
/// CSVファイルとして保存します（外部でのデータ可視化や分析用途）。
pub fn save_embeddings(tensor: &Tensor, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let (vocab_size, d_model) = tensor.dims2()?;
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;

    let mut file = File::create(filename)?;
    for i in 0..vocab_size {
        let start = i * d_model;
        let end = start + d_model;
        let row = &data[start..end];
        let row_strings: Vec<String> = row.iter().map(|v| v.to_string()).collect();
        writeln!(file, "{}", row_strings.join(","))?;
    }
    println!("Embeddings saved to {}", filename);
    Ok(())
}

/// ユーザーからのテキスト入力を受け取り、AIとしてテキスト応答を返すチャット機能。
pub fn chat(
    transformer: &Transformer,
    device: &Device,
    tokenizer: &mut CharTokenizer,
    text: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    println!("Chat Input: {}", text);
    let log_msg = format!("thinking: processing chat input '{}'", text);
    append_thinking_log(&log_msg)?;

    // 入力テキストをトークン化
    let input_tokens = tokenizer.encode(text);
    if input_tokens.is_empty() {
        return Ok(String::new());
    }
    // テキスト生成を実行（ここでは最大10トークン生成）
    let generated = generate_text(transformer, device, input_tokens, 10)?;

    // 生成されたトークンをテキストに復元して出力
    let output_text = tokenizer.decode(&generated);

    println!("Chat Output: {}", output_text);
    let log_msg = format!("thinking: generated chat output '{}'", output_text);
    append_thinking_log(&log_msg)?;

    Ok(output_text)
}

/// 特定の単語(word)とその意味(meaning_text)のペアをモデルに学習させます。
/// 「{word}の意味は{meaning_text}です。」という文として学習(小規模なファインチューニング)します。
pub fn meaning(
    transformer: &Transformer,
    varmap: &VarMap,
    device: &Device,
    tokenizer: &mut CharTokenizer,
    word: &str,
    meaning_text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let text = format!("{}の意味は{}です。", word, meaning_text);
    println!("Meaning: Learning '{}' -> '{}'", word, meaning_text);
    let log_msg = format!(
        "thinking: learning meaning of word '{}' as '{}'",
        word, meaning_text
    );
    append_thinking_log(&log_msg)?;

    // 学習用のテキストを作成してトークン化
    let tokens = tokenizer.encode(&text);
    if tokens.len() < 2 {
        return Ok(());
    }

    // 最後のトークンを予測するように入力と正解(ターゲット)を1つずらして作成
    let input_tokens = &tokens[0..tokens.len() - 1];
    let target_tokens = &tokens[1..tokens.len()];

    let input_tensor = Tensor::from_vec(input_tokens.to_vec(), (1, input_tokens.len()), device)?;
    let target_tensor = Tensor::from_vec(target_tokens.to_vec(), (1, target_tokens.len()), device)?;

    // OptimizerとしてAdamWを使用 (学習率 0.001)
    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.001)?;

    // 5ステップだけ学習させて重みを更新します
    for _ in 0..5 {
        let logits = transformer.forward(&input_tensor)?.squeeze(0)?;
        let target = target_tensor.squeeze(0)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &target)?;
        opt.backward_step(&loss)?;
    }

    println!("Meaning of '{}' has been learned.", word);
    Ok(())
}

/// 任意の入力(input)と出力(output)のペアをモデルに学習させます。
/// 基本的な仕組みや目的は `meaning` 関数と同様です。
pub fn training(
    transformer: &Transformer,
    varmap: &VarMap,
    device: &Device,
    tokenizer: &mut CharTokenizer,
    input: &str,
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Training: Input='{}', Output='{}'", input, output);
    let log_msg = format!(
        "thinking: training on input '{}' and output '{}'",
        input, output
    );
    append_thinking_log(&log_msg)?;

    // 入力と出力をスペースで繋げてトークン化
    let text = format!("{} {}", input, output);
    let tokens = tokenizer.encode(&text);

    if tokens.len() < 2 {
        return Ok(());
    }

    // 入力と正解(ターゲット)の系列を作成
    let input_tokens = &tokens[0..tokens.len() - 1];
    let target_tokens = &tokens[1..tokens.len()];

    let input_tensor = Tensor::from_vec(input_tokens.to_vec(), (1, input_tokens.len()), device)?;
    let target_tensor = Tensor::from_vec(target_tokens.to_vec(), (1, target_tokens.len()), device)?;

    // Optimizerを用意して学習を実行
    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.001)?;

    // 5ステップの学習を実施
    for _ in 0..5 {
        let logits = transformer.forward(&input_tensor)?.squeeze(0)?;
        let target = target_tensor.squeeze(0)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &target)?;
        opt.backward_step(&loss)?;
    }

    println!("Training completed for the given input-output pair.");
    Ok(())
}
