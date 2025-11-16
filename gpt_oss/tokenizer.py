import tiktoken

def get_tokenizer():
    """Get the o200k_harmony tokenizer used by gpt-oss-20b.
    
    Tries to use openai-harmony package if available, otherwise falls back
    to manual construction using tiktoken.
    """
    # Try to use openai-harmony package if available
    try:
        from openai_harmony import load_harmony_encoding, HarmonyEncodingName
        # Use the correct Harmony encoding for GPT-OSS
        return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    except ImportError:
        pass
    except AttributeError:
        # Fallback if HarmonyEncodingName doesn't have HARMONY_GPT_OSS
        pass
    
    # Fallback: manually construct Harmony tokenizer
    o200k_base = tiktoken.get_encoding("o200k_base")
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
            **o200k_base._special_tokens,
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        } | {
            f"<|reserved_{i}|>": i for i in range(200013, 201088)
        },
    )
    return tokenizer
