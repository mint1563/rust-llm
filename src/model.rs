use candle_core::{Device, Result, Tensor};
use candle_nn::{linear, layer_norm, Linear, LayerNorm, Module, VarBuilder};

/// Transformerモデルのハイパーパラメータを保持する設定構造体
#[derive(Debug, Clone)]
pub struct Config {
    /// d_model: モデルの隠れ層の次元数（トークンの埋め込みベクトルのサイズ）
    pub d_model: usize,
    /// n_head: Multi-Head Attentionのヘッド数
    pub n_head: usize,
    /// n_layer: Transformerブロック（層）を重ねる数
    pub n_layer: usize,
    /// vocab_size: 扱う語彙（トークン）の総数
    pub vocab_size: usize,
}

impl Config {
    /// デフォルトの設定値でConfigを作成します
    pub fn new() -> Self {
        Self {
            d_model: 256,
            n_head: 8,
            n_layer: 4,
            vocab_size: 10000,
        }
    }
}

/// 因果的自己注意機構 (Causal Self-Attention)
/// 未来のトークン情報を参照しないようにマスクをかけるAttentionです（Decoder専用）
#[derive(Debug)]
struct CausalSelfAttention {
    /// Query, Key, Valueを同時に計算するための線形層
    c_attn: Linear,
    /// Attention適用後の出力を元の次元に投影する線形層
    c_proj: Linear,
    n_head: usize,
    d_model: usize,
}

impl CausalSelfAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d_model = cfg.d_model;
        // 入力次元 d_model に対して、Query, Key, Valueの3つ分(d_model * 3)を一度に計算します
        let c_attn = linear(d_model, d_model * 3, vb.pp("c_attn"))?;
        let c_proj = linear(d_model, d_model, vb.pp("c_proj"))?;
        Ok(Self {
            c_attn,
            c_proj,
            n_head: cfg.n_head,
            d_model,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x の形状: (バッチサイズ, シーケンス長, モデル次元数)
        let (b_sz, seq_len, d_model) = x.dims3()?;
        let head_dim = self.d_model / self.n_head;

        // 1. Q, K, V を同時に計算
        // 形状: (b_sz, seq_len, d_model * 3)
        let qkv = self.c_attn.forward(x)?; 

        // 2. Q, K, V に分割 (それぞれ次元数は d_model)
        let q = qkv.narrow(2, 0, d_model)?;
        let k = qkv.narrow(2, d_model, d_model)?;
        let v = qkv.narrow(2, d_model * 2, d_model)?;

        // 3. Multi-Head用に形状を変換し、ヘッドの次元を前に持ってくる
        // (b_sz, seq_len, n_head, head_dim) -> (b_sz, n_head, seq_len, head_dim)
        // メモリ上に連続して配置(contiguous)させ、後続の行列積を高速・安全に行います
        let q = q.reshape((b_sz, seq_len, self.n_head, head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b_sz, seq_len, self.n_head, head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b_sz, seq_len, self.n_head, head_dim))?.transpose(1, 2)?.contiguous()?;

        // 4. Scaled Dot-Product Attention の計算: Attention Score = Q * K^T / sqrt(d_k)
        // Kを転置: (b_sz, n_head, head_dim, seq_len)
        let k_t = k.transpose(2, 3)?.contiguous()?;
        // QとK^Tの行列積: (b_sz, n_head, seq_len, seq_len)
        let att = (q.matmul(&k_t)? / (head_dim as f64).sqrt())?;
        
        // 5. Causal Mask（因果マスク）の適用
        // 未来のトークンへのAttention Scoreをマイナス無限大にして、Softmaxで0になるようにします
        let mask = Self::get_causal_mask(seq_len, x.device())?;
        let att = att.broadcast_add(&mask)?;

        // 6. Softmax関数を適用して確率の重みに変換
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;

        // 7. Attentionの重みとValue(V)を掛け合わせる
        // 形状: (b_sz, n_head, seq_len, head_dim)
        let y = att.matmul(&v)?;

        // 8. Multi-Headの結果を結合し、元の形状に戻す
        // (b_sz, n_head, seq_len, head_dim) -> (b_sz, seq_len, n_head, head_dim) -> (b_sz, seq_len, d_model)
        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, d_model))?.contiguous()?;

        // 9. 最後の線形層を通す
        self.c_proj.forward(&y)
    }

    /// 未来のトークンを見ないためのマスクを生成する関数
    /// 例（シーケンス長3の場合）:
    /// [[ 0.0, -inf, -inf],
    ///  [ 0.0,  0.0, -inf],
    ///  [ 0.0,  0.0,  0.0]]
    fn get_causal_mask(t: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<_> = (0..t).flat_map(|i| {
            (0..t).map(move |j| {
                if j > i {
                    f32::NEG_INFINITY // 未来のトークンには -∞ を設定
                } else {
                    0f32              // 過去・現在のトークンは 0 を設定
                }
            })
        }).collect();
        let mask = Tensor::from_vec(mask, (t, t), device)?;
        // ブロードキャスト計算ができるように次元を追加 (1, 1, t, t)
        mask.reshape((1, 1, t, t))
    }
}

/// Feed Forward Network (多層パーセプトロン / MLP)
/// 各トークンの表現を独立して非線形変換します
#[derive(Debug)]
struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // 一般的に中間層の次元はモデル次元の4倍にします
        let hidden_dim = cfg.d_model * 4;
        let c_fc = linear(cfg.d_model, hidden_dim, vb.pp("c_fc"))?;
        let c_proj = linear(hidden_dim, cfg.d_model, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 第1層
        let x = self.c_fc.forward(x)?;
        // GELU活性化関数（最近のTransformerでよく使われます）
        let x = x.gelu()?;
        // 第2層（元の次元に戻す）
        self.c_proj.forward(&x)
    }
}

/// Transformer ブロック (1層分)
/// Attention機構とMLP、それに正規化層(LayerNorm)と残差接続(Residual Connection)をまとめます
#[derive(Debug)]
struct Block {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: MLP,
}

impl Block {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln_1 = layer_norm(cfg.d_model, 1e-5, vb.pp("ln_1"))?;
        let attn = CausalSelfAttention::new(cfg, vb.pp("attn"))?;
        let ln_2 = layer_norm(cfg.d_model, 1e-5, vb.pp("ln_2"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        Ok(Self { ln_1, attn, ln_2, mlp })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-LN（処理の前に正規化する）アーキテクチャを採用
        
        // 1. Attentionブロック + 残差接続
        let residual = x;
        let x = self.ln_1.forward(x)?;
        let x = self.attn.forward(&x)?;
        let x = (x + residual)?; // 入力(residual)をそのまま足し合わせることで勾配消失を防ぐ

        // 2. MLPブロック + 残差接続
        let residual = &x;
        let x = self.ln_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;

        Ok(x)
    }
}

/// GPT型のTransformer Decoder本体
/// すべてのコンポーネントを束ねて、トークン列を受け取り次の単語の予測（Logits）を出力します
#[derive(Debug)]
pub struct Transformer {
    /// Token Embedding: 単語IDをベクトル表現に変換する層
    wte: candle_nn::Embedding,
    /// Positional Embedding: トークンの位置情報をベクトル表現に変換する層
    wpe: candle_nn::Embedding,
    /// 複数スタックされたTransformerブロック
    blocks: Vec<Block>,
    /// 最終層のLayer Normalization
    ln_f: LayerNorm,
    /// Language Model Head: ベクトル表現を元の語彙サイズの確率分布(Logits)に戻す線形層
    lm_head: Linear,
}

impl Transformer {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // 単語ID -> ベクトル
        let wte = candle_nn::embedding(cfg.vocab_size, cfg.d_model, vb.pp("wte"))?;
        
        // 位置 -> ベクトル (最大シーケンス長を1024と仮定)
        let max_seq_len = 1024; 
        let wpe = candle_nn::embedding(max_seq_len, cfg.d_model, vb.pp("wpe"))?;
        
        // 指定された層の数だけブロックを作成
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            blocks.push(Block::new(cfg, vb.pp(&format!("h.{}", i)))?);
        }
        
        let ln_f = layer_norm(cfg.d_model, 1e-5, vb.pp("ln_f"))?;
        let lm_head = linear(cfg.d_model, cfg.vocab_size, vb.pp("lm_head"))?;
        
        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x の形状: (バッチサイズ, シーケンス長) ※値は単語IDの整数
        let (_b_sz, seq_len) = x.dims2()?;
        let device = x.device();
        
        // 0 から シーケンス長-1 までの位置IDを作成
        let pos = Tensor::arange(0u32, seq_len as u32, device)?;
        let pos = pos.unsqueeze(0)?; // 形状を (1, seq_len) に変更
        
        // 単語ベクトルと位置ベクトルの取得
        let tok_emb = self.wte.forward(x)?;     // 形状: (b_sz, seq_len, d_model)
        let pos_emb = self.wpe.forward(&pos)?;  // 形状: (1, seq_len, d_model)
        
        // 2つのベクトルを足し合わせて入力表現を完成させる
        let mut x = tok_emb.broadcast_add(&pos_emb)?;
        
        // 全てのTransformerブロックを順番に通過させる
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        
        // 最終正規化層を適用
        let x = self.ln_f.forward(&x)?;
        
        // LM Headで元の語彙のサイズに次元を広げ、次の単語の予測スコア(Logits)を取得
        // 出力形状: (b_sz, seq_len, vocab_size)
        let logits = self.lm_head.forward(&x)?;
        
        Ok(logits)
    }

    /// 単語の埋め込み(Embedding)ベクトルを取得します
    pub fn get_wte_tensor(&self) -> &Tensor {
        self.wte.embeddings()
    }
}