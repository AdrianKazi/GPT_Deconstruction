# Model 0 - One Attention Head

## Imports

```python

```


```python

```

		Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
		WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



```python
# GPT Model Deconstruction — Quick Preview

**Notebook:** `GPT_Model_Deconstruction.ipynb` contains full, line-by-line explanations and runnable code.  
**README (this file):** curated quick preview — key concepts, formulas and representative visual results (no code). See the notebook for full derivations and implementation details.

---

## Overview

This project deconstructs a minimal GPT-style model (one attention head) to illustrate how embeddings, projections, normalization and attention interact. The notebook contains the full step-by-step narrative; this README surfaces the most important conceptual points and visual diagnostics for quick inspection.

## Core equations

The attention projections and causal softmax used in the experiments:

$$Q = XW_Q,\\quad K = XW_K,\\quad V = XW_V$$

$$A = \\operatorname{softmax}\\left(\\frac{QK^{\\top}}{\\sqrt{d_k}} + M\\right)$$

$$H = AV$$

The causal mask $M$ enforces autoregressive attention:
$$M_{ij}=\\begin{cases}0,& j\\le i\\\\-\\infty,& j>i\\end{cases}$$

## Key findings & visuals

Below are selected conceptual notes and figures extracted from the notebook. Explanations that refer to the internal code cell-by-cell have been omitted.

### Layer normalization

Layer normalization standardizes the embedding vectors along the feature dimension, producing distributions with mean near 0 and unit variance — a stabilizing effect visible in the figure below.

![LayerNorm effect](figures/notebook_preview_md_22_1.png)

### Q/K/V projections

Linear projections reproject token embeddings into the attention subspace; the plots show representative projection statistics and distributions.

![Projection diagnostics](figures/notebook_preview_md_33_0.png)

### Additional diagnostics

Representative visualizations used throughout the analysis:

![Fig 1](figures/notebook_preview_md_100_0.png) ![Fig 2](figures/notebook_preview_md_102_0.png) ![Fig 3](figures/notebook_preview_md_113_0.png)

![Fig 4](figures/notebook_preview_md_115_0.png) ![Fig 5](figures/notebook_preview_md_157_0.png) ![Fig 6](figures/notebook_preview_md_167_0.png)

![Fig 7](figures/notebook_preview_md_169_0.png) ![Fig 8](figures/notebook_preview_md_172_0.png) ![Fig 9](figures/notebook_preview_md_217_0.png)

![Fig 10](figures/notebook_preview_md_218_0.png) ![Fig 11](figures/notebook_preview_md_220_0.png) ![Fig 12](figures/notebook_preview_md_243_0.png)

![Fig 13](figures/notebook_preview_md_246_0.png) ![Fig 14](figures/notebook_preview_md_250_0.png) ![Fig 15](figures/notebook_preview_md_41_0.png)

![Fig 16](figures/notebook_preview_md_45_0.png) ![Fig 17](figures/notebook_preview_md_49_0.png) ![Fig 18](figures/notebook_preview_md_58_1.png)

![Fig 19](figures/notebook_preview_md_61_0.png) ![Fig 20](figures/notebook_preview_md_67_0.png) ![Fig 21](figures/notebook_preview_md_77_0.png)

![Fig 22](figures/notebook_preview_md_82_0.png)

---

## Usage

- Open `GPT_Model_Deconstruction.ipynb` to run experiments and read full explanations.  
- This README is a concise, publication-style preview designed for quick inspection by reviewers and collaborators.

If you'd like, I will add short captions to each figure, a brief abstract at the top, and a `LICENSE` file (e.g., MIT). Please confirm which items to add.
		<>:15: SyntaxWarning: invalid escape sequence '\s'
		<>:14: SyntaxWarning: invalid escape sequence '\m'
		<>:14: SyntaxWarning: invalid escape sequence '\s'
		<>:15: SyntaxWarning: invalid escape sequence '\m'
		<>:15: SyntaxWarning: invalid escape sequence '\s'
		/tmp/ipykernel_11065/1343297862.py:14: SyntaxWarning: invalid escape sequence '\m'
			plt.plot(pre_dist,  'b-o', label = f'Pre- Norm embed_dim | $\mu = {pre_mean:>10.1f}$ | $\sigma = {pre_std:.1f}$')
		/tmp/ipykernel_11065/1343297862.py:14: SyntaxWarning: invalid escape sequence '\s'
			plt.plot(pre_dist,  'b-o', label = f'Pre- Norm embed_dim | $\mu = {pre_mean:>10.1f}$ | $\sigma = {pre_std:.1f}$')
		/tmp/ipykernel_11065/1343297862.py:15: SyntaxWarning: invalid escape sequence '\m'
			plt.plot(post_dist, 'r-o', label = f'Pre- Norm embed_dim | $\mu = {post_mean:>10.1f}$ | $\sigma = {post_std:.1f}$')
		/tmp/ipykernel_11065/1343297862.py:15: SyntaxWarning: invalid escape sequence '\s'
			plt.plot(post_dist, 'r-o', label = f'Pre- Norm embed_dim | $\mu = {post_mean:>10.1f}$ | $\sigma = {post_std:.1f}$')



![png](figures/notebook_preview_md_22_1.png)


Normalization changes last dimension into $\mu = 0$ and $\sigma = 1$ what we can experience. So, last dimension which is `embed_dim` gets above parameters, in simpler words, each each token, in each batch, has vector of `embed_dim` length, this vector for each token gets normalized.

One question is why we need to pass `embed_dim` in `__init__` in `nn.LayerNorm(embed_dim)`? `nn.LayerNorm()` either way always takes last dimenion and normalizes on last dimension, then why we need to give a shape of last dimension in `__init__`? Well, that's because we want to give all parameters in `__init__`, that's how after just class instantion we can check model parameters:


```python

```

		torch.Size([50257, 64])
		torch.Size([8, 64])
		torch.Size([64])
		torch.Size([64])
		torch.Size([64, 64])
		torch.Size([64, 64])
		torch.Size([64, 64])
		torch.Size([64, 64])
		torch.Size([64])
		torch.Size([50257, 64])
		torch.Size([50257])


Developers theoretically could make LayerNorm infer the last dimension during the first forward pass, but then parameters like LayerNorm.weight and LayerNorm.bias would not exist immediately after __init__.

That would make model.parameters(), state_dict(), and optimizer initialization less straightforward before the first forward pass.

#### `k`,`q`,`v`

First, $W_Q$, $W_K$, $W_V$ are the raw learnable weight matrices.
Q, K, V are the projected matrices after multiplying X by those weights:

$$Q = XW_Q$$
$$K = XW_K$$
$$V = XW_V$$

in code we named $W_Q$, $W_K$, $W_V$ by `query`, `key`, `value` variables.


```python

```



		(torch.Size([10, 8, 64]), torch.Size([10, 8, 64]), torch.Size([10, 8, 64]))


`nn.Linear(x)` is just a dot product `x@[embed_dim,embed_dim].T`


```python

```



		(torch.Size([10, 8, 64]), torch.Size([10, 8, 64]), torch.Size([10, 8, 64]))


We get exactly same results.


```python

```


![png](figures/notebook_preview_md_33_0.png)


Since $W,Q,K$ are of shapes `[embed_dim,embed_dim]` and $x$ is of shape `[batch_size, num_tokens, embed_dim]` (taken from last step) we do the multiplication on last 2 dimensions matrices, so for x `[num_tokens, embed_dim]` and for $W,Q,K$ `[embed_dim, embed_dim]` after matrix multiplication we receive shape of `[num_tokens, embed_dim]` which is broadcasted through all batches, so eventually we receive `[batch_size, num_tokens, embed_dim]`.

#### `qk`


```python

```

Notes:
# GPT_Deconstruction
