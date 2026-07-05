# GPT Model Deconstruction

This repository presents a full deconstruction of one GPT-style autoregressive language model. The notebook is written as a step-by-step build-up: first the smallest possible attention mechanism is opened and inspected, then the same logic is extended with residual connections, MLP, multiple transformer blocks, multi-head attention, GPT-scale dimensions, GPU/CPU placement, and finally a direct inspection of the original pretrained GPT-2.

The code sections in the notebook are not independent final architectures. They are checkpoints used to break one GPT architecture into understandable parts. Each checkpoint adds a little more of the same final GPT structure, so the reader can see exactly what changes in tensors, shapes, probabilities, representations, and generated text.

Full implementation: [GPT_Model_Deconstruction.ipynb](./GPT_Model_Deconstruction.ipynb)

This README is the professional, code-free overview of the notebook. The notebook contains the full runnable implementation, all code, tensor checks, plots, outputs, and GPT-2 inspection.

## Abstract

The project explains how a GPT-style model transforms token ids into next-token probabilities. It starts from token and positional embeddings, then follows the residual stream through layer normalization, Q/K/V projections, causal masking, softmax attention, value mixing, output projection, MLP transformation, multiple transformer blocks, multi-head splitting, and vocabulary projection.

The main idea is simple but deep: every token starts as an embedding vector, receives position information, looks only into the past through causal self-attention, mixes information from previous tokens, gets updated through attention and MLP residual branches, and finally becomes a vector of logits over the whole vocabulary. Generation then takes the last-token logits, applies temperature, samples the next token, appends it to the context, and repeats the same process.

The central attention mechanism is:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)
$$

$$
H = AV
$$

The causal mask is the part that makes the model autoregressive:

$$
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

In simple words: the token at a given position can use itself and previous tokens, but it cannot see future tokens.

## Repository Contents

| Path | Purpose |
| --- | --- |
| [GPT_Model_Deconstruction.ipynb](./GPT_Model_Deconstruction.ipynb) | Full implementation: code, explanations, tensor shapes, plots, outputs, and GPT-2 inspection |
| [README.md](./README.md) | Code-free scientific overview of the whole notebook |
| [figures/](./figures/) | Extracted notebook plots used in this README |

## Full GPT Pipeline

| Stage in the full GPT | What the notebook shows |
| --- | --- |
| Tokenization | Text is converted into GPT-2 token ids using the GPT-2 tokenizer |
| Token embeddings | Token ids become dense vectors with shape `[batch_size, num_tokens, embed_dim]` |
| Positional embeddings | Position vectors are added so the model knows token order |
| Residual stream | The model keeps a main representation stream that attention and MLP update instead of replacing |
| LayerNorm | Each token vector is normalized across `embed_dim` before major transformations |
| Q/K/V projection | The residual stream is projected into query, key, and value representations |
| Attention scores | `QK^T` produces token-to-token scores with shape `[batch_size, num_tokens, num_tokens]` |
| Scaling | Scores are divided by `sqrt(embed_dim)` or `sqrt(head_dim)` to keep softmax stable |
| Causal mask | Future positions are replaced with `-inf`, so softmax gives them probability zero |
| Attention weights | Softmax turns scores into a probability distribution over visible tokens |
| Value mixing | Attention weights multiply `V`, producing contextual token representations |
| Output projection | The attention result is rewritten back into model space |
| Residual addition | Attention output is added back to the residual stream |
| MLP | The token representation expands to `4 * embed_dim`, passes through GELU, and contracts back |
| Transformer depth | The same block is repeated many times so representations are refined gradually |
| Multi-head attention | `embed_dim` is split into independent heads, each attention head works in its own subspace |
| Vocabulary logits | Final hidden states are projected to `n_vocab = 50257` logits |
| Sampling | Last-token logits are converted into probabilities and sampled with temperature |
| GPT-2 inspection | The handmade structure is compared with the original pretrained GPT-2 architecture |

## Detailed Deconstruction

### Embeddings and Residual Stream

The notebook begins with integer token ids. These ids are not meaningful to the model by themselves, so they are passed through token embeddings. Each token receives a vector of length `embed_dim`. Positional embeddings are then added to token embeddings, because attention alone does not know the order of the sequence.

The important shape is:

| Representation | Shape in the first inspection |
| --- | --- |
| Token ids | `[10, 8]` |
| Token embeddings | `[10, 8, 64]` |
| Positional embeddings | `[8, 64]` |
| Token plus position representation | `[10, 8, 64]` |

The positional embedding is broadcasted through every batch. Eventually, every token in every batch has a vector of length `embed_dim`. This is the beginning of the residual stream.

### Layer Normalization

LayerNorm normalizes each token vector along the last dimension, which is `embed_dim`. The notebook explicitly asks why `embed_dim` must be passed into `LayerNorm`. The reason is that LayerNorm has learnable weight and bias parameters, so the model must know their shape already in initialization. That makes `model.parameters()`, `state_dict()`, and optimizer initialization straightforward before the first forward pass.

Conceptually, normalization changes each token representation toward mean `0` and standard deviation `1`. This stabilizes the stream before attention and MLP updates.

![LayerNorm effect](figures/notebook_preview_md_22_1.png)

The first plot appears exactly at the LayerNorm step. It shows the pre-normalization and post-normalization embedding distribution for a selected token, so it belongs here before Q/K/V projection.

### Q, K, V

The notebook then takes the normalized stream and creates three projected matrices:

| Object | Meaning |
| --- | --- |
| `W_Q`, `W_K`, `W_V` | Raw learnable projection matrices |
| `Q` | Query: what the current token is looking for |
| `K` | Key: what each token offers as a matching signal |
| `V` | Value: the information that will actually be mixed |

The key point is that Q, K, and V keep the same batch and sequence dimensions while changing the representation space. In the first attention inspection, all three have shape `[10, 8, 64]`.

![Q/K/V projection diagnostics](figures/notebook_preview_md_33_0.png)

This plot belongs directly after the Q/K/V projection because it shows the projected matrices and the projection weights before any attention score is computed.

### Attention Scores and Scaling

The notebook multiplies `Q` by transposed `K`. For one batch item, this means a `[8, 64]` matrix multiplied by a `[64, 8]` matrix, producing an `[8, 8]` token-to-token score matrix. Across all batches, the shape is `[10, 8, 8]`.

![Raw attention score matrices](figures/notebook_preview_md_41_0.png)

The raw `QK^T` plot is placed here because it is the first visual view of token-to-token interaction, before scaling and before the causal mask.

These raw scores are then scaled by the square root of the embedding dimension. This matters because dot products grow with dimensionality. Without scaling, the softmax can become too sharp too early.

![Scaled attention scores](figures/notebook_preview_md_45_0.png)

The scaled-score plot compares the raw and scaled score ranges, so it belongs immediately after the scaling explanation.

### Causal Mask

The mask simulates time. At the first token, the model can only see the first token. At later positions, it can see more previous tokens. Future tokens are not allowed, because text generation is causal.

![Causal mask](figures/notebook_preview_md_49_0.png)

This plot shows the lower-triangular past mask and the score matrix after future positions are blocked. It is the visual proof that the model is causal.

The notebook also demonstrates why masked positions need `-inf`, not `0`. Softmax over a value of `0` still gives that entry probability. Softmax over `-inf` gives zero probability, which is exactly what causal language modeling needs.

### Softmax Attention and Value Mixing

After masking, softmax turns each row of attention scores into a probability distribution over previous tokens. This distribution says which previous tokens matter for building the current token representation.

![Masked softmax attention](figures/notebook_preview_md_58_1.png)

The masked-softmax plot is placed here because it comes after the causal mask and before multiplying by `V`.

![Attention distribution close-up](figures/notebook_preview_md_61_0.png)

The close-up plot shows one selected position and makes the same point in a more detailed way: only causally valid positions receive attention probability.

Then attention weights multiply `V`. Before multiplication, the attention matrix has shape `[batch_size, num_tokens, num_tokens]`. After multiplying by `V`, the output returns to `[batch_size, num_tokens, embed_dim]`.

In simple words: for each token position, attention attaches back a mixed value vector of length `embed_dim`.

![Value mixing](figures/notebook_preview_md_67_0.png)

This plot belongs at the exact moment where `attention_weights @ V` is explained. It shows how attention weights select and mix value vectors into a contextual representation.

### Output Projection and Vocabulary Projection

After attention, the output projection rewrites the mixed value vector back into model space.

![Output projection comparison](figures/notebook_preview_md_77_0.png)

The output-projection plot is placed before vocabulary logits because it still operates inside the hidden representation space.

The final vocabulary projection maps the token representation from `embed_dim` to `n_vocab`. In the GPT-2 tokenizer setup, `n_vocab = 50257`, so logits have shape `[batch_size, num_tokens, 50257]`.

![Vocabulary projection](figures/notebook_preview_md_82_0.png)

This plot shows the transition from hidden states into vocabulary logits, so it closes the forward-pass part before generation starts.

### Temperature and Sampling

The notebook then uses the final sequence position for generation. This is important: next-token generation does not sample from every token position, but from the logits at the last position of the current context.

![Last-token logits heatmap](figures/notebook_preview_md_100_0.png)

The heatmap is placed at the first generation step because it shows logits taken from the last token position.

![Last-token logits close-up](figures/notebook_preview_md_102_0.png)

The close-up plot stays directly after the full heatmap because it zooms into the same next-token distribution.

Generation applies softmax to the last-token logits divided by temperature. Higher temperature spreads probability mass across more tokens, so sampled text becomes more variable. Lower temperature concentrates probability mass around the strongest tokens, so generation becomes more predictable.

![Temperature sampling distribution](figures/notebook_preview_md_113_0.png)

This plot belongs with the temperature explanation because it shows how higher temperature spreads probability mass across many tokens.

![Temperature 1 sampling distribution](figures/notebook_preview_md_115_0.png)

This plot follows immediately because it shows the same sampling mechanism with a more concentrated distribution.

The notebook uses multinomial sampling, meaning the selected token does not have to be the highest-probability token. It is sampled according to the probability distribution. After sampling, the next token is appended to the token sequence. The same process repeats for every new generated token.

### Full Transformer Block

The notebook then wraps attention into the full transformer block. The block has two main update branches:

| Branch | Role |
| --- | --- |
| Attention branch | Lets tokens communicate with previous tokens |
| MLP branch | Transforms each token representation independently through expansion, GELU, and contraction |

Both branches use residual connections. This is the key transformer idea in the notebook: attention and MLP do not replace the stream. They produce updates that are added back to it.

![Residual attention update](figures/notebook_preview_md_157_0.png)

This plot appears at the residual addition step: the attention output is not the new state by itself; it is an update added back into the stream.

The transformer block flow is:

| Step | Meaning |
| --- | --- |
| Pre-attention LayerNorm | Normalize residual stream before attention |
| Attention | Compute contextual update |
| First residual addition | Add attention output back to the stream |
| Pre-MLP LayerNorm | Normalize before MLP |
| MLP expansion | Increase dimensionality to `4 * embed_dim` |
| GELU | Add nonlinearity |
| MLP contraction | Return to `embed_dim` |
| Second residual addition | Add MLP output back to the stream |

The notebook emphasizes why the MLP expands the dimension. The expansion gives the representation more space, or different angles, to transform features. The geometric plot shows that some structures are hard to separate in lower dimension but easier after adding another dimension.

![MLP geometric intuition](figures/notebook_preview_md_167_0.png)

This plot is placed directly under the MLP expansion explanation because it visualizes the idea of giving the representation a larger feature space.

![Nonlinear separation intuition](figures/notebook_preview_md_169_0.png)

The separation plot follows the expansion plot because it explains why added dimensions can make patterns easier to separate.

![GELU activation](figures/notebook_preview_md_172_0.png)

The GELU plot is placed after the expansion/separation intuition because GELU is the nonlinearity applied inside the MLP before contraction back to `embed_dim`.

### Depth

The transformer block is then repeated sequentially. Each block keeps the same tensor shape, but each block changes the hidden representation slightly. Because of residual connections, updates are incremental, and the accumulated effect across depth gradually refines the representation.

![Single-block representation change](figures/notebook_preview_md_217_0.png)

The first depth plot shows the effect of one transformer block, so it comes before the full 12-block visualization.

![Representations across 12 blocks](figures/notebook_preview_md_218_0.png)

This plot belongs next because it shows how the residual stream changes across all 12 sequential blocks.

![Block-to-block differences](figures/notebook_preview_md_220_0.png)

The difference plot closes the depth section because it shows incremental updates between consecutive blocks.

### Multi-Head Attention

The notebook then replaces one large attention operation with multi-head attention. Instead of keeping `embed_dim = 256` as one attention space, it splits it into `num_heads = 8` and `head_dim = 32`.

The important transformation is:

| Before split | After split |
| --- | --- |
| `[batch_size, num_tokens, embed_dim]` | `[batch_size, num_heads, num_tokens, head_dim]` |

![Q/K/V before head split](figures/notebook_preview_md_243_0.png)

This plot is placed before the split because Q, K, and V still live in the full embedding dimension.

![Q/K/V after head split](figures/notebook_preview_md_246_0.png)

This plot follows immediately because it shows the same Q, K, and V tensors after splitting `embed_dim` into `num_heads` and `head_dim`.

This means the model computes several smaller attentions instead of one huge attention block. Each head can focus on a different representation subspace. After attention, heads are transposed and merged back into `embed_dim`.

![Multi-head merge](figures/notebook_preview_md_250_0.png)

The merge plot closes the multi-head section because it shows independent heads being joined back into one `embed_dim` representation.

The notebook's conclusion is direct: multi-head attention gives the model multiple independent attention views over the same sequence.

### GPT-Scale Configuration

After the mechanism is understood, the notebook scales the implementation toward GPT-style settings:

| Setting | Value captured in notebook |
| --- | --- |
| Batch size | `8` |
| Context length | `1024` |
| Embedding dimension | `768` |
| Vocabulary size | `50257` |
| Transformer block count | `12` |
| Multi-head attention output | `[8, 1024, 768]` |
| Transformer block output | `[8, 1024, 768]` |
| Generated tensor shape | `[8, 1044]` |

The notebook also documents device placement. Moving `Model().to(device)` moves parameters created in initialization, but tensors newly created inside `forward`, such as positional index tensors, must also be created on or moved to the correct device.

Captured generation timing shows that CPU generation for this long-context configuration takes roughly three minutes in the notebook run. The notebook notes that GPU is much faster, and the difference grows with more generated tokens.

### Original GPT-2 Inspection

The final notebook section loads Hugging Face's pretrained GPT-2 and compares it with the implementation built step by step. The inspection confirms the same architectural structure:

| GPT-2 component | Captured shape or observation |
| --- | --- |
| Token embedding matrix | `[50257, 768]` |
| Positional embedding matrix | `[1024, 768]` |
| Transformer stack | 12 blocks |
| Attention module | GPT-2 attention with combined Q/K/V projection |
| Combined Q/K/V projection size | `2304`, equal to `3 * 768` |
| Final generation prompt | `Hello, how are you today?` |
| Generated output | Coherent English continuation from pretrained GPT-2 |

This closes the loop: the notebook first builds the mechanism from scratch, then inspects the real GPT-2 implementation and shows that the same pieces are present.

## Captured Outputs

| Output area | What it demonstrates |
| --- | --- |
| GPT-2 tokenizer loading | The notebook uses the GPT-2 tokenizer and may show Hugging Face unauthenticated request warnings |
| Parameter shapes | Embeddings, LayerNorm weights, attention projections, output projections, and vocabulary projection parameters are printed and inspected |
| Masking demonstration | `0` is not enough for masking because softmax still assigns probability; `-inf` removes probability |
| Untrained generation | Early generated samples are noisy or repetitive because architecture alone is not training |
| Temperature behavior | Higher temperature increases variability; lower temperature makes the strongest logits dominate |
| Long-context generation | Full GPT-style settings produce `[8, 1044]` generated token tensors |
| GPT-2 summary | Pretrained GPT-2 loads with expected transformer blocks and parameter structure |
| GPT-2 generation | Pretrained GPT-2 produces coherent text, unlike the untrained intermediate checkpoints |

## Technical Note

The notebook has been cleaned for GitHub rendering. Large widget state metadata and Colab-specific widget outputs were removed, while notebook cells, text outputs, image outputs, and all extracted figures were preserved.

For the complete implementation, open [GPT_Model_Deconstruction.ipynb](./GPT_Model_Deconstruction.ipynb).
