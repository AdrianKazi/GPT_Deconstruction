# Model 0 - One Attention Head

## Imports

```python

```


```python

```

		Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
		WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



```python

```



		['SPECIAL_TOKENS_ATTRIBUTES',
		 '__annotations__',
		 '__bool__',
		 '__call__',
		 '__class__',
		 '__delattr__',
		 '__dict__',
		 '__dir__',
		 '__doc__',
		 '__eq__',
		 '__format__',
		 '__ge__',
		 '__getattr__',
		 '__getattribute__',
		 '__getstate__',
		 '__gt__',
		 '__hash__',
		 '__init__',
		 '__init_subclass__',
		 '__le__',
		 '__len__',
		 '__lt__',
		 '__module__',
		 '__ne__',
		 '__new__',
		 '__reduce__',
		 '__reduce_ex__',
		 '__repr__',
		 '__setattr__',
		 '__sizeof__',
		 '__str__',
		 '__subclasshook__',
		 '__weakref__',
		 '_add_bos_token',
		 '_add_eos_token',
		 '_add_tokens',
		 '_added_tokens_decoder',
		 '_added_tokens_encoder',
		 '_auto_class',
		 '_convert_encoding',
		 '_convert_id_to_token',
		 '_convert_token_to_id_with_added_voc',
		 '_decode',
		 '_encode_plus',
		 '_eventual_warn_about_too_long_sequence',
		 '_extra_special_tokens',
		 '_from_pretrained',
		 '_get_files_timestamps',
		 '_get_padding_truncation_strategies',
		 '_in_target_context_manager',
		 '_merges',
		 '_pad',
		 '_pad_token_type_id',
		 '_patch_mistral_regex',
		 '_post_init',
		 '_processor_class',
		 '_save_pretrained',
		 '_set_model_specific_special_tokens',
		 '_set_processor_class',
		 '_should_update_post_processor',
		 '_special_tokens_map',
		 '_tokenizer',
		 '_upload_modified_files',
		 '_vocab',
		 'add_bos_token',
		 'add_eos_token',
		 'add_prefix_space',
		 'add_special_tokens',
		 'add_tokens',
		 'added_tokens_decoder',
		 'added_tokens_encoder',
		 'all_special_ids',
		 'all_special_tokens',
		 'apply_chat_template',
		 'backend',
		 'backend_tokenizer',
		 'batch_decode',
		 'can_save_slow_tokenizer',
		 'chat_template',
		 'clean_up_tokenization',
		 'clean_up_tokenization_spaces',
		 'clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output',
		 'convert_added_tokens',
		 'convert_ids_to_tokens',
		 'convert_to_native_format',
		 'convert_tokens_to_ids',
		 'convert_tokens_to_string',
		 'decode',
		 'decoder',
		 'deprecation_warnings',
		 'encode',
		 'encode_message_with_chat_template',
		 'files_loaded',
		 'from_pretrained',
		 'get_added_vocab',
		 'get_chat_template',
		 'get_special_tokens_mask',
		 'get_vocab',
		 'init_inputs',
		 'init_kwargs',
		 'is_fast',
		 'max_len_sentences_pair',
		 'max_len_single_sentence',
		 'model',
		 'model_input_names',
		 'model_max_length',
		 'name_or_path',
		 'num_special_tokens_to_add',
		 'pad',
		 'pad_token_type_id',
		 'padding_side',
		 'parse_response',
		 'pretrained_vocab_files_map',
		 'push_to_hub',
		 'register_for_auto_class',
		 'response_schema',
		 'save_chat_templates',
		 'save_pretrained',
		 'save_vocabulary',
		 'set_truncation_and_padding',
		 'slow_tokenizer_class',
		 'special_tokens_map',
		 'split_special_tokens',
		 'tokenize',
		 'train_new_from_iterator',
		 'truncation_side',
		 'update_post_processor',
		 'verbose',
		 'vocab',
		 'vocab_file',
		 'vocab_files_names',
		 'vocab_size']



## Hyperparameters


```python

```

## Model

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
A =
\operatorname{softmax}
\left(
\frac{QK^\\top}{\\sqrt{d_k}} + M
\right)
$$

$$
H = AV
$$

$$
\\operatorname{CausalAttention}(Q,K,V)
=
H W_O
=
\\operatorname{softmax}
\left(
\frac{QK^\\top}{\\sqrt{d_k}} + M
\right)
V W_O
$$

$$
M_{ij}
=
\begin{cases}
0, & j \le i \\
-\\infty, & j > i
\end{cases}
$$


```python

```

## Deconstruction

### forward()

#### `tok_embed`, `pos_embed`


```python

```



		torch.Size([10, 8, 64])



`nn.Embedding` accepts 2 values: `(num_embeddings, embedding_dim)`. In our example, `num_embeddings` is `n_vocab`, i.e. the input to `nn.Embedding` contains integer indices, and each number must be lower than `num_embeddings` (in our example, lower than `n_vocab`). `embedding_dim` is the length of the vector assigned to each index. For example, `tok_x` has shape `[batch_size, num_tokens]`, meaning each element inside `tok_x` is a token ID from `0` to `n_vocab - 1`. The outcome is that for every batch and every token, we get an embedding vector of length `embedding_dim`; therefore, `token_embeddings` has shape `[batch_size, num_tokens, embed_dim]`.


```python

```


		torch.Size([8, 64])


It works same for position embeddings, here we pass arange of length `num_tokens` taken from `tok_x`. That causes, than each index of arange get the embedding vector.

#### `layer_norm_1`


```python

```

```python

```


```python

```

		<>:14: SyntaxWarning: invalid escape sequence '\m'
		<>:14: SyntaxWarning: invalid escape sequence '\s'
		<>:15: SyntaxWarning: invalid escape sequence '\m'
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



![png](notebook_preview_md_files/notebook_preview_md_22_1.png)


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


![png](notebook_preview_md_files/notebook_preview_md_33_0.png)


Since $W,Q,K$ are of shapes `[embed_dim,embed_dim]` and $x$ is of shape `[batch_size, num_tokens, embed_dim]` (taken from last step) we do the multiplication on last 2 dimensions matrices, so for x `[num_tokens, embed_dim]` and for $W,Q,K$ `[embed_dim, embed_dim]` after matrix multiplication we receive shape of `[num_tokens, embed_dim]` which is broadcasted through all batches, so eventually we receive `[batch_size, num_tokens, embed_dim]`.

#### `qk`


```python

```

Notes:
# GPT_Deconstruction
