# GPT Model Deconstruction

This repo deconstructs one GPT-style model step by step. The notebook starts with the smallest useful piece, one attention head, and then keeps adding what a real GPT needs: residual connections, MLP, many transformer blocks, multi-head attention, GPT-scale dimensions, device handling, and finally inspection of the original pretrained GPT-2.

The sections in the notebook are not meant to be separate final models. They are checkpoints. Each checkpoint adds one more piece, so it is clear what changes in shapes, tensors, probabilities, plots, and generated text.

Full implementation: [GPT_Model_Deconstruction.ipynb](./GPT_Model_Deconstruction.ipynb)

This README is the clean version without code. The notebook has the full implementation, all tensor checks, outputs, plots, and GPT-2 inspection.

## Main Idea

The whole thing is about one question: how does GPT take token ids and turn them into probabilities for the next token?

The flow is:

1. Take token ids.
2. Turn them into token embeddings.
3. Add positional embeddings.
4. Normalize the representation.
5. Create Q, K, and V.
6. Compute attention scores.
7. Hide the future with a causal mask.
8. Turn scores into attention probabilities.
9. Mix value vectors.
10. Add attention and MLP updates back into the residual stream.
11. Project the final representation to vocabulary logits.
12. Sample the next token and repeat.

The attention formula is:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)
$$

$$
H = AV
$$

The mask is what makes it causal:

$$
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

In simple words: the model can look at the current token and the past, but it cannot look into the future.

## Repository Contents

| Path | What it is |
| --- | --- |
| [GPT_Model_Deconstruction.ipynb](./GPT_Model_Deconstruction.ipynb) | Full notebook with code, explanations, shapes, plots, outputs, and GPT-2 inspection |
| [README.md](./README.md) | Same story, but without code |
| [figures/](./figures/) | Plots extracted from the notebook |

## Full GPT Flow

| Step | What happens |
| --- | --- |
| Tokenization | Text becomes GPT-2 token ids |
| Token embeddings | Token ids become vectors shaped `[batch_size, num_tokens, embed_dim]` |
| Positional embeddings | Position vectors are added, so the model knows token order |
| Residual stream | This is the main representation that attention and MLP keep updating |
| LayerNorm | Each token vector is normalized across `embed_dim` |
| Q/K/V | The representation is projected into query, key, and value |
| `QK^T` | We get token-to-token attention scores |
| Scaling | Scores are divided by `sqrt(embed_dim)` or `sqrt(head_dim)` |
| Causal mask | Future tokens are changed to `-inf` |
| Softmax | Scores become attention probabilities |
| `attention @ V` | Attention attaches back a mixed value vector |
| Output projection | The mixed value vector is rewritten back into model space |
| Residual add | We add the attention update back to `x` |
| MLP | We expand to `4 * embed_dim`, apply GELU, and shrink back |
| Blocks | The same transformer block is repeated many times |
| Multi-head | One big attention is split into many smaller attentions |
| Vocabulary logits | Hidden states become logits over `n_vocab = 50257` |
| Sampling | Last-token logits become probabilities, and we sample the next token |
| GPT-2 inspection | We check that the original GPT-2 has the same pieces |

## Deconstruction

### Embeddings and Residual Stream

We start with integer token ids. These ids are just indexes, so by themselves they do not carry the kind of information the model can work with. First, we pass them through token embeddings. Then we add positional embeddings, because attention does not know token order by itself.

The important shapes are:

| Representation | Shape |
| --- | --- |
| Token ids | `[10, 8]` |
| Token embeddings | `[10, 8, 64]` |
| Positional embeddings | `[8, 64]` |
| Token plus position representation | `[10, 8, 64]` |

`pos_embed` is added to the last two dimensions of `tok_embed` and broadcasted through every batch. Eventually we get `[batch_size, num_tokens, embed_dim]`. This is the start of the residual stream.

### LayerNorm

LayerNorm changes the last dimension into something close to mean `0` and standard deviation `1`. In simpler words, each token in each batch has a vector of length `embed_dim`, and this vector gets normalized.

One question is why we pass `embed_dim` into `LayerNorm`. LayerNorm normalizes the last dimension anyway, so why give it the shape? Because LayerNorm has parameters: weight and bias. The model needs to know their shape already in `__init__`, so `model.parameters()`, `state_dict()`, and optimizer setup work before the first forward pass.

![LayerNorm effect](figures/notebook_preview_md_22_1.png)

As we can see, this is exactly the normalization step. The plot compares the token vector before and after LayerNorm.

### Q, K, V

First, `W_Q`, `W_K`, and `W_V` are raw learnable weight matrices. Q, K, and V are the projected matrices after multiplying `x` by those weights.

| Object | Meaning |
| --- | --- |
| `Q` | What the current token is looking for |
| `K` | What each token offers for matching |
| `V` | The information that will actually be mixed |

Q, K, and V keep the batch and token dimensions. In the first attention walkthrough, they all have shape `[10, 8, 64]`.

![Q/K/V projection diagnostics](figures/notebook_preview_md_33_0.png)

This plot belongs here because it shows Q, K, V and the projection weights before we compute any attention scores.

### `QK^T` and Scaling

Now we multiply `Q` with transposed `K`. For one batch, it is `[8, 64] @ [64, 8]`, so we get `[8, 8]`. Broadcasted through all 10 batches, it becomes `[10, 8, 8]`.

![Raw attention score matrices](figures/notebook_preview_md_41_0.png)

This is the first moment where tokens start interacting with other tokens. These are still raw scores, before scaling and before masking.

Then we divide by the square root of the dimension. We do this because dot products get larger when `embed_dim` gets larger. If we do not scale, softmax can become too sharp.

![Scaled attention scores](figures/notebook_preview_md_45_0.png)

As we can see, scaling compresses the scores before softmax.

### Causal Mask

`past_mask` simulates time. At the top we have one token, and going forward we add more tokens. The thing is that the model is causal, so it needs to see only the past. Training text has future tokens in it, but we hide them from the model.

![Causal mask](figures/notebook_preview_md_49_0.png)

This plot shows the lower-triangular mask and the attention scores after future positions are blocked.

Why `-inf` and not `0`? Because softmax over `0` is not zero. If we want the probability to be exactly zero, we need negative infinity.

### Softmax Attention

After masking, softmax turns scores into a distribution over previous tokens. The word “probability” can be a little misleading here. It is better to think about it as importance weights: how much information should we take from each previous token?

![Masked softmax attention](figures/notebook_preview_md_58_1.png)

Here the masked scores become attention weights.

![Attention distribution close-up](figures/notebook_preview_md_61_0.png)

This is the same idea zoomed in for one selected position. Only the allowed past positions can get attention.

### `attention @ V`

Then we multiply the attention weights by `V`.

Before multiplication we have `[batch_size, num_tokens, num_tokens]`. After multiplication with `V`, we receive `[batch_size, num_tokens, embed_dim]`.

In simple words: for each token position, attention attaches back a mixed value vector of length `embed_dim`.

![Value mixing](figures/notebook_preview_md_67_0.png)

For a selected position, if one previous token has weight `0.9`, most of the new context vector comes from that token's value vector. Tokens with weight near `0` contribute almost nothing.

### Output Projection and Vocabulary Logits

After attention, every token gets back a mixed value vector. Then `W0` rewrites this mixed information back into model space.

![Output projection comparison](figures/notebook_preview_md_77_0.png)

This is still inside the hidden representation space. We are not at vocabulary logits yet.

Then we project from `embed_dim` to `n_vocab`. In this notebook, `n_vocab = 50257`, because we use the GPT-2 tokenizer.

![Vocabulary projection](figures/notebook_preview_md_82_0.png)

Now each token is represented by a vector of vocabulary logits.

### Generation, Last Token, and Temperature

For generation, we care about the last token position. The output has shape `[batch_size, seq_len, n_vocab]`, but for next-token prediction we take `x[:, -1, :]`.

![Last-token logits heatmap](figures/notebook_preview_md_100_0.png)

This shows logits for the last token position.

![Last-token logits close-up](figures/notebook_preview_md_102_0.png)

Here we zoom into the same next-token distribution.

Then we apply softmax with temperature. Higher temperature means more variability. Lower temperature means the strongest logits dominate more.

![Temperature sampling distribution](figures/notebook_preview_md_113_0.png)

With higher temperature, more tokens become possible.

![Temperature 1 sampling distribution](figures/notebook_preview_md_115_0.png)

With temperature near `1`, the distribution is more concentrated.

Then we use multinomial sampling. The chosen token does not need to be the token with the highest probability. It is sampled from the distribution. After that, we append it to `tok_x`, and the generation window moves forward.

### Full Transformer Block

Now we wrap attention into a full transformer block. No magic here: it is attention, residual connection, MLP, and another residual connection.

The main idea is that attention does not replace `x`. Attention produces an update, and we add that update back to the residual stream. Same thing later with MLP.

![Residual attention update](figures/notebook_preview_md_157_0.png)

As we can see, the attention output changes the representation, but the shape stays the same.

The block flow is:

| Step | What happens |
| --- | --- |
| LayerNorm before attention | Normalize `x` |
| Attention | Compute token-to-token update |
| Residual add | `x = residual + attn_out` |
| LayerNorm before MLP | Normalize updated `x` |
| MLP expansion | Go from `embed_dim` to `4 * embed_dim` |
| GELU | Add non-linearity |
| MLP contraction | Go back to `embed_dim` |
| Residual add | Add MLP update back to `x` |

### MLP

The MLP is basically:

`embed_dim -> 4 * embed_dim -> GELU -> embed_dim`

Okay, but why do we do this? By adding new dimensions, we give the model different angles of view on the representation. The exact `4 * embed_dim` expansion is a convention that works well.

![MLP geometric intuition](figures/notebook_preview_md_167_0.png)

This plot shows the idea of expanding into a larger representation space.

![Nonlinear separation intuition](figures/notebook_preview_md_169_0.png)

By adding another dimension, some patterns become easier to separate. That is the intuition behind giving the representation more space.

Then we apply GELU. Without a non-linearity, the whole MLP would still be just a linear transformation.

![GELU activation](figures/notebook_preview_md_172_0.png)

GELU filters the expanded representation before we shrink it back to `embed_dim`.

### Many Transformer Blocks

Now we stack transformer blocks. We simply wrapped Attention and MLP in one class, and then we repeat it 12 times.

Each block keeps the same shape, but changes the representation a bit. Because we use residual connections, every block updates `x` instead of destroying it and starting over.

![Single-block representation change](figures/notebook_preview_md_217_0.png)

This shows what one block does.

![Representations across 12 blocks](figures/notebook_preview_md_218_0.png)

Here we see the representation changing across all 12 blocks.

![Block-to-block differences](figures/notebook_preview_md_220_0.png)

And here we see differences between consecutive blocks. The updates are incremental, but they accumulate.

### Multi-Head Attention

Now instead of one big attention over `embed_dim = 256`, we split it into:

`num_heads = 8`, `head_dim = 32`

So instead of one huge attention block, we have 8 smaller attention heads.

![Q/K/V before head split](figures/notebook_preview_md_243_0.png)

Before splitting, Q, K, and V still live in the full embedding dimension.

![Q/K/V after head split](figures/notebook_preview_md_246_0.png)

After splitting, each token has 8 smaller vectors, one for each head.

Eventually, we merge `[num_heads, head_dim]` back into `embed_dim`. The transpositions are mostly there because PyTorch expects tensors in `[batch_size, num_heads, tok_nums, head_dim]`.

![Multi-head merge](figures/notebook_preview_md_250_0.png)

So the simple conclusion is this: instead of computing one attention mechanism over the whole representation, we split it into several attention heads and let each head look at the sequence in its own subspace.

### GPT-Scale Version

After the mechanism is clear, the notebook scales it toward GPT-style settings.

| Setting | Value |
| --- | --- |
| Batch size | `8` |
| Context length | `1024` |
| Embedding dimension | `768` |
| Vocabulary size | `50257` |
| Transformer blocks | `12` |
| Multi-head attention output | `[8, 1024, 768]` |
| Transformer block output | `[8, 1024, 768]` |
| Generated tensor | `[8, 1044]` |

Device movement is also simple. We move the model and input tensors to the device, but tensors created inside `forward`, like `torch.arange`, also need to be created on the same device.

The notebook run also shows that CPU generation for the long context is slow, around a few minutes. GPU is much faster, and the difference grows when we generate more tokens.

### Original GPT-2 Inspection

At the end, the notebook loads the original pretrained GPT-2 and checks what is inside.

| GPT-2 part | What we see |
| --- | --- |
| Token embedding matrix | `[50257, 768]` |
| Positional embedding matrix | `[1024, 768]` |
| Transformer blocks | 12 |
| Attention | GPT-2 attention with combined Q/K/V projection |
| Combined Q/K/V size | `2304`, which is `3 * 768` |
| Prompt | `Hello, how are you today?` |
| Output | Coherent English continuation |

This is the final check: first we build the mechanism step by step, then we inspect real GPT-2 and see the same pieces.

## Captured Outputs

| Output | What it shows |
| --- | --- |
| GPT-2 tokenizer loading | The notebook uses GPT-2 tokenizer; Hugging Face warnings can appear |
| Parameter shapes | Embeddings, LayerNorm, attention projections, output projection, and vocabulary projection are inspected |
| Masking test | `0` is not enough for masking; `-inf` is needed |
| Untrained generation | Architecture alone does not produce good text |
| Temperature | Higher temperature adds variability; lower temperature is more predictable |
| GPT-scale generation | The long-context setup produces `[8, 1044]` generated tensors |
| GPT-2 summary | Pretrained GPT-2 has the expected transformer structure |
| GPT-2 generation | Pretrained GPT-2 generates coherent text |

## Technical Note

The notebook has been cleaned for GitHub rendering. Large widget state metadata and Colab-specific widget outputs were removed, while notebook cells, text outputs, image outputs, and all extracted figures were preserved.

For the full implementation, open [GPT_Model_Deconstruction.ipynb](./GPT_Model_Deconstruction.ipynb).

## Reference

Models are based on Mike X Cohen course and editted, deconstruction and plots are made by myself.
