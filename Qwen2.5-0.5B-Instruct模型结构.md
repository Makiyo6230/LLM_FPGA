# Qwen2.5-0.5B-Instruct 模型结构

## 1. Config 摘要

```text
model_type: qwen2
architectures: ['Qwen2ForCausalLM']
torch_dtype: bfloat16
vocab_size: 151936
hidden_size: 896
intermediate_size: 4864
num_hidden_layers: 24
num_attention_heads: 14
num_key_value_heads: 2
head_dim: 64
gqa_repeat: 7
rms_norm_eps: 1e-06
rope_theta: 1000000.0
max_position_embeddings: 32768
attention_bias: None
tie_word_embeddings: True
```

## 2. 逻辑模块结构

```text
Qwen2ForCausalLM
  model.embed_tokens.weight                  [151936, 896]
  model.layers[0..23]
    input_layernorm.weight                   [896]
    self_attn
      q_proj.weight                          [896, 896]
      k_proj.weight                          [128, 896]
      v_proj.weight                          [128, 896]
      o_proj.weight                          [896, 896]
    post_attention_layernorm.weight          [896]
    mlp
      gate_proj.weight                       [4864, 896]
      up_proj.weight                         [4864, 896]
      down_proj.weight                       [896, 4864]
  model.norm.weight                          [896]
  lm_head.weight                             [151936, 896]
```

注意：`config.json` 中 `tie_word_embeddings=True`，因此实际 `model.safetensors` 里没有单独的 `lm_head.weight` tensor；lm head 使用 `model.embed_tokens.weight` 作为输出词表投影权重。

## 3. 权重文件概况

```text
safetensors files: 1
  model.safetensors
bin files: 0
```

实际权重 tensor 数量：`290`。

## 4. FPGA 第 0 层 bring-up 常用权重

```text
model.layers.0.input_layernorm.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.q_proj.bias
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.k_proj.bias
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.v_proj.bias
model.layers.0.self_attn.o_proj.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.norm.weight
lm_head.weight
```

其中 `lm_head.weight` 是逻辑名；当前模型权重绑定，实际读取时使用 `model.embed_tokens.weight`。

## 5. 完整权重 tensor 名称与 shape

|    # | 权重名                                              | Shape             | DType    |    参数量 | 文件                  |
| ---: | --------------------------------------------------- | ----------------- | -------- | --------: | --------------------- |
| 0001 | `model.embed_tokens.weight`                       | `(151936, 896)` | `BF16` | 136134656 | `model.safetensors` |
| 0002 | `model.layers.0.input_layernorm.weight`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0003 | `model.layers.0.mlp.down_proj.weight`             | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0004 | `model.layers.0.mlp.gate_proj.weight`             | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0005 | `model.layers.0.mlp.up_proj.weight`               | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0006 | `model.layers.0.post_attention_layernorm.weight`  | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0007 | `model.layers.0.self_attn.k_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0008 | `model.layers.0.self_attn.k_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0009 | `model.layers.0.self_attn.o_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0010 | `model.layers.0.self_attn.q_proj.bias`            | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0011 | `model.layers.0.self_attn.q_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0012 | `model.layers.0.self_attn.v_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0013 | `model.layers.0.self_attn.v_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0014 | `model.layers.1.input_layernorm.weight`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0015 | `model.layers.1.mlp.down_proj.weight`             | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0016 | `model.layers.1.mlp.gate_proj.weight`             | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0017 | `model.layers.1.mlp.up_proj.weight`               | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0018 | `model.layers.1.post_attention_layernorm.weight`  | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0019 | `model.layers.1.self_attn.k_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0020 | `model.layers.1.self_attn.k_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0021 | `model.layers.1.self_attn.o_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0022 | `model.layers.1.self_attn.q_proj.bias`            | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0023 | `model.layers.1.self_attn.q_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0024 | `model.layers.1.self_attn.v_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0025 | `model.layers.1.self_attn.v_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0026 | `model.layers.10.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0027 | `model.layers.10.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0028 | `model.layers.10.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0029 | `model.layers.10.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0030 | `model.layers.10.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0031 | `model.layers.10.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0032 | `model.layers.10.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0033 | `model.layers.10.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0034 | `model.layers.10.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0035 | `model.layers.10.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0036 | `model.layers.10.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0037 | `model.layers.10.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0038 | `model.layers.11.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0039 | `model.layers.11.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0040 | `model.layers.11.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0041 | `model.layers.11.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0042 | `model.layers.11.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0043 | `model.layers.11.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0044 | `model.layers.11.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0045 | `model.layers.11.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0046 | `model.layers.11.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0047 | `model.layers.11.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0048 | `model.layers.11.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0049 | `model.layers.11.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0050 | `model.layers.12.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0051 | `model.layers.12.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0052 | `model.layers.12.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0053 | `model.layers.12.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0054 | `model.layers.12.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0055 | `model.layers.12.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0056 | `model.layers.12.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0057 | `model.layers.12.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0058 | `model.layers.12.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0059 | `model.layers.12.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0060 | `model.layers.12.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0061 | `model.layers.12.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0062 | `model.layers.13.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0063 | `model.layers.13.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0064 | `model.layers.13.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0065 | `model.layers.13.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0066 | `model.layers.13.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0067 | `model.layers.13.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0068 | `model.layers.13.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0069 | `model.layers.13.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0070 | `model.layers.13.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0071 | `model.layers.13.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0072 | `model.layers.13.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0073 | `model.layers.13.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0074 | `model.layers.14.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0075 | `model.layers.14.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0076 | `model.layers.14.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0077 | `model.layers.14.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0078 | `model.layers.14.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0079 | `model.layers.14.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0080 | `model.layers.14.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0081 | `model.layers.14.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0082 | `model.layers.14.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0083 | `model.layers.14.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0084 | `model.layers.14.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0085 | `model.layers.14.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0086 | `model.layers.15.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0087 | `model.layers.15.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0088 | `model.layers.15.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0089 | `model.layers.15.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0090 | `model.layers.15.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0091 | `model.layers.15.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0092 | `model.layers.15.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0093 | `model.layers.15.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0094 | `model.layers.15.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0095 | `model.layers.15.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0096 | `model.layers.15.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0097 | `model.layers.15.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0098 | `model.layers.16.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0099 | `model.layers.16.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0100 | `model.layers.16.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0101 | `model.layers.16.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0102 | `model.layers.16.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0103 | `model.layers.16.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0104 | `model.layers.16.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0105 | `model.layers.16.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0106 | `model.layers.16.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0107 | `model.layers.16.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0108 | `model.layers.16.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0109 | `model.layers.16.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0110 | `model.layers.17.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0111 | `model.layers.17.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0112 | `model.layers.17.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0113 | `model.layers.17.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0114 | `model.layers.17.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0115 | `model.layers.17.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0116 | `model.layers.17.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0117 | `model.layers.17.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0118 | `model.layers.17.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0119 | `model.layers.17.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0120 | `model.layers.17.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0121 | `model.layers.17.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0122 | `model.layers.18.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0123 | `model.layers.18.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0124 | `model.layers.18.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0125 | `model.layers.18.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0126 | `model.layers.18.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0127 | `model.layers.18.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0128 | `model.layers.18.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0129 | `model.layers.18.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0130 | `model.layers.18.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0131 | `model.layers.18.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0132 | `model.layers.18.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0133 | `model.layers.18.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0134 | `model.layers.19.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0135 | `model.layers.19.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0136 | `model.layers.19.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0137 | `model.layers.19.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0138 | `model.layers.19.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0139 | `model.layers.19.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0140 | `model.layers.19.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0141 | `model.layers.19.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0142 | `model.layers.19.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0143 | `model.layers.19.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0144 | `model.layers.19.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0145 | `model.layers.19.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0146 | `model.layers.2.input_layernorm.weight`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0147 | `model.layers.2.mlp.down_proj.weight`             | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0148 | `model.layers.2.mlp.gate_proj.weight`             | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0149 | `model.layers.2.mlp.up_proj.weight`               | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0150 | `model.layers.2.post_attention_layernorm.weight`  | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0151 | `model.layers.2.self_attn.k_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0152 | `model.layers.2.self_attn.k_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0153 | `model.layers.2.self_attn.o_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0154 | `model.layers.2.self_attn.q_proj.bias`            | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0155 | `model.layers.2.self_attn.q_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0156 | `model.layers.2.self_attn.v_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0157 | `model.layers.2.self_attn.v_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0158 | `model.layers.20.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0159 | `model.layers.20.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0160 | `model.layers.20.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0161 | `model.layers.20.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0162 | `model.layers.20.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0163 | `model.layers.20.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0164 | `model.layers.20.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0165 | `model.layers.20.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0166 | `model.layers.20.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0167 | `model.layers.20.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0168 | `model.layers.20.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0169 | `model.layers.20.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0170 | `model.layers.21.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0171 | `model.layers.21.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0172 | `model.layers.21.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0173 | `model.layers.21.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0174 | `model.layers.21.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0175 | `model.layers.21.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0176 | `model.layers.21.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0177 | `model.layers.21.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0178 | `model.layers.21.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0179 | `model.layers.21.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0180 | `model.layers.21.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0181 | `model.layers.21.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0182 | `model.layers.22.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0183 | `model.layers.22.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0184 | `model.layers.22.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0185 | `model.layers.22.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0186 | `model.layers.22.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0187 | `model.layers.22.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0188 | `model.layers.22.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0189 | `model.layers.22.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0190 | `model.layers.22.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0191 | `model.layers.22.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0192 | `model.layers.22.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0193 | `model.layers.22.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0194 | `model.layers.23.input_layernorm.weight`          | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0195 | `model.layers.23.mlp.down_proj.weight`            | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0196 | `model.layers.23.mlp.gate_proj.weight`            | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0197 | `model.layers.23.mlp.up_proj.weight`              | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0198 | `model.layers.23.post_attention_layernorm.weight` | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0199 | `model.layers.23.self_attn.k_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0200 | `model.layers.23.self_attn.k_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0201 | `model.layers.23.self_attn.o_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0202 | `model.layers.23.self_attn.q_proj.bias`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0203 | `model.layers.23.self_attn.q_proj.weight`         | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0204 | `model.layers.23.self_attn.v_proj.bias`           | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0205 | `model.layers.23.self_attn.v_proj.weight`         | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0206 | `model.layers.3.input_layernorm.weight`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0207 | `model.layers.3.mlp.down_proj.weight`             | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0208 | `model.layers.3.mlp.gate_proj.weight`             | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0209 | `model.layers.3.mlp.up_proj.weight`               | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0210 | `model.layers.3.post_attention_layernorm.weight`  | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0211 | `model.layers.3.self_attn.k_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0212 | `model.layers.3.self_attn.k_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0213 | `model.layers.3.self_attn.o_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0214 | `model.layers.3.self_attn.q_proj.bias`            | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0215 | `model.layers.3.self_attn.q_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0216 | `model.layers.3.self_attn.v_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0217 | `model.layers.3.self_attn.v_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0218 | `model.layers.4.input_layernorm.weight`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0219 | `model.layers.4.mlp.down_proj.weight`             | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0220 | `model.layers.4.mlp.gate_proj.weight`             | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0221 | `model.layers.4.mlp.up_proj.weight`               | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0222 | `model.layers.4.post_attention_layernorm.weight`  | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0223 | `model.layers.4.self_attn.k_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0224 | `model.layers.4.self_attn.k_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0225 | `model.layers.4.self_attn.o_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0226 | `model.layers.4.self_attn.q_proj.bias`            | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0227 | `model.layers.4.self_attn.q_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0228 | `model.layers.4.self_attn.v_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0229 | `model.layers.4.self_attn.v_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0230 | `model.layers.5.input_layernorm.weight`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0231 | `model.layers.5.mlp.down_proj.weight`             | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0232 | `model.layers.5.mlp.gate_proj.weight`             | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0233 | `model.layers.5.mlp.up_proj.weight`               | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0234 | `model.layers.5.post_attention_layernorm.weight`  | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0235 | `model.layers.5.self_attn.k_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0236 | `model.layers.5.self_attn.k_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0237 | `model.layers.5.self_attn.o_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0238 | `model.layers.5.self_attn.q_proj.bias`            | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0239 | `model.layers.5.self_attn.q_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0240 | `model.layers.5.self_attn.v_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0241 | `model.layers.5.self_attn.v_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0242 | `model.layers.6.input_layernorm.weight`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0243 | `model.layers.6.mlp.down_proj.weight`             | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0244 | `model.layers.6.mlp.gate_proj.weight`             | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0245 | `model.layers.6.mlp.up_proj.weight`               | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0246 | `model.layers.6.post_attention_layernorm.weight`  | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0247 | `model.layers.6.self_attn.k_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0248 | `model.layers.6.self_attn.k_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0249 | `model.layers.6.self_attn.o_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0250 | `model.layers.6.self_attn.q_proj.bias`            | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0251 | `model.layers.6.self_attn.q_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0252 | `model.layers.6.self_attn.v_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0253 | `model.layers.6.self_attn.v_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0254 | `model.layers.7.input_layernorm.weight`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0255 | `model.layers.7.mlp.down_proj.weight`             | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0256 | `model.layers.7.mlp.gate_proj.weight`             | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0257 | `model.layers.7.mlp.up_proj.weight`               | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0258 | `model.layers.7.post_attention_layernorm.weight`  | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0259 | `model.layers.7.self_attn.k_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0260 | `model.layers.7.self_attn.k_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0261 | `model.layers.7.self_attn.o_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0262 | `model.layers.7.self_attn.q_proj.bias`            | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0263 | `model.layers.7.self_attn.q_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0264 | `model.layers.7.self_attn.v_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0265 | `model.layers.7.self_attn.v_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0266 | `model.layers.8.input_layernorm.weight`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0267 | `model.layers.8.mlp.down_proj.weight`             | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0268 | `model.layers.8.mlp.gate_proj.weight`             | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0269 | `model.layers.8.mlp.up_proj.weight`               | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0270 | `model.layers.8.post_attention_layernorm.weight`  | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0271 | `model.layers.8.self_attn.k_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0272 | `model.layers.8.self_attn.k_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0273 | `model.layers.8.self_attn.o_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0274 | `model.layers.8.self_attn.q_proj.bias`            | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0275 | `model.layers.8.self_attn.q_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0276 | `model.layers.8.self_attn.v_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0277 | `model.layers.8.self_attn.v_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0278 | `model.layers.9.input_layernorm.weight`           | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0279 | `model.layers.9.mlp.down_proj.weight`             | `(896, 4864)`   | `BF16` |   4358144 | `model.safetensors` |
| 0280 | `model.layers.9.mlp.gate_proj.weight`             | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0281 | `model.layers.9.mlp.up_proj.weight`               | `(4864, 896)`   | `BF16` |   4358144 | `model.safetensors` |
| 0282 | `model.layers.9.post_attention_layernorm.weight`  | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0283 | `model.layers.9.self_attn.k_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0284 | `model.layers.9.self_attn.k_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0285 | `model.layers.9.self_attn.o_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0286 | `model.layers.9.self_attn.q_proj.bias`            | `(896)`         | `BF16` |       896 | `model.safetensors` |
| 0287 | `model.layers.9.self_attn.q_proj.weight`          | `(896, 896)`    | `BF16` |    802816 | `model.safetensors` |
| 0288 | `model.layers.9.self_attn.v_proj.bias`            | `(128)`         | `BF16` |       128 | `model.safetensors` |
| 0289 | `model.layers.9.self_attn.v_proj.weight`          | `(128, 896)`    | `BF16` |    114688 | `model.safetensors` |
| 0290 | `model.norm.weight`                               | `(896)`         | `BF16` |       896 | `model.safetensors` |

## 6. 与 FPGA 验证的关系

```text
activation .npy: 模块输入/输出样例，用于 golden 对比。
model.safetensors: 模型固定权重，带参数模块必须读取对应 tensor。
```

例如第 0 层 RMSNorm 验证需要：

```text
输入 activation:
  qwen25_0p5b_instruct_full_generation_trace/00_prefill_full_prompt/layer_00_transformer_block/attention_rmsnorm_input.npy

权重:
  model.layers.0.input_layernorm.weight

golden 输出:
  qwen25_0p5b_instruct_full_generation_trace/00_prefill_full_prompt/layer_00_transformer_block/attention_rmsnorm_output.npy
```
