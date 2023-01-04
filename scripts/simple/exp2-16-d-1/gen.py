default = {
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": False,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 256,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 8,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "type_vocab_size": 2,
  "use_cache": False,
  "vocab_size": 10000,
  "override_output_vocab_size": 10000,
  "tie_word_embeddings": False
}

import json
for size in [768, 384, 96]:
    for layer in [8, 4, 1]:
        for head in [12, 6, 1]:
            default["hidden_size"] = size
            default["intermediate_size"] = size * 4
            default["num_hidden_layers"] = layer
            default["num_attention_heads"] = head
            with open(f"config{size}-{layer}-{head}.json", "wt") as f:
                json.dump(default, f, indent=2)