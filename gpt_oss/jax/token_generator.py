"""TokenGenerator for JAX backend.

Provides a unified interface for token generation that matches other backends.
"""

import jax
import jax.numpy as jnp
from pathlib import Path
from typing import List, Optional, Iterator, Tuple

from .model import Transformer
from .config import ModelConfig
from .inference import generate
from .loader_orbax import OrbaxWeightLoader, load_config_from_orbax
from .loader_safetensors import WeightLoader


def _detect_checkpoint_format(checkpoint_path: str) -> str:
    """Detect checkpoint format (Orbax or SafeTensors).
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        'orbax' or 'safetensors'
    """
    path = Path(checkpoint_path)
    
    # Check for Orbax checkpoint: should have a "0" subdirectory with state
    if (path / "0").exists():
        state_path = path / "0" / "state"
        if state_path.exists() and (state_path / "_METADATA").exists():
            return "orbax"
    
    # Check for SafeTensors: should have .safetensors files
    if list(path.glob("*.safetensors")):
        return "safetensors"
    
    # Default to SafeTensors if unclear
    return "safetensors"


def _load_config_from_checkpoint(checkpoint_path: str, format: str) -> ModelConfig:
    """Load model configuration from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        format: Checkpoint format ('orbax' or 'safetensors')
        
    Returns:
        ModelConfig instance
    """
    if format == "orbax":
        # Orbax checkpoints don't include config.json, use hardcoded defaults from loader
        config_dict = load_config_from_orbax(checkpoint_path)
        # Use the config values returned by the loader (they match the checkpoint)
        return ModelConfig(
            num_hidden_layers=config_dict["num_hidden_layers"],
            hidden_size=config_dict["hidden_size"],
            head_dim=config_dict["head_dim"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_key_value_heads=config_dict["num_key_value_heads"],
            sliding_window=config_dict["sliding_window"],
            intermediate_size=config_dict["intermediate_size"],
            num_experts=config_dict["num_experts"],
            experts_per_token=config_dict["experts_per_token"],
            vocab_size=config_dict["vocab_size"],
            swiglu_limit=config_dict["swiglu_limit"],
            rope_theta=config_dict["rope_theta"],
            rope_scaling_factor=config_dict["rope_scaling_factor"],
            rope_ntk_alpha=config_dict["rope_ntk_alpha"],
            rope_ntk_beta=config_dict["rope_ntk_beta"],
            initial_context_length=config_dict["initial_context_length"],
        )
    else:
        # SafeTensors: try to load config.json, fallback to defaults
        config_path = Path(checkpoint_path) / "config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return ModelConfig(
                num_hidden_layers=config_dict.get("num_hidden_layers", 36),
                hidden_size=config_dict.get("hidden_size", 2880),
                head_dim=config_dict.get("head_dim", 64),
                num_attention_heads=config_dict.get("num_attention_heads", 64),
                num_key_value_heads=config_dict.get("num_key_value_heads", 8),
                sliding_window=config_dict.get("sliding_window", 128),
                intermediate_size=config_dict.get("intermediate_size", 2880),
                num_experts=config_dict.get("num_experts", 128),
                experts_per_token=config_dict.get("experts_per_token", 4),
                vocab_size=config_dict.get("vocab_size", 201088),
                swiglu_limit=config_dict.get("swiglu_limit", 7.0),
                rope_theta=config_dict.get("rope_theta", 150000.0),
                rope_scaling_factor=config_dict.get("rope_scaling_factor", 32.0),
                rope_ntk_alpha=config_dict.get("rope_ntk_alpha", 1.0),
                rope_ntk_beta=config_dict.get("rope_ntk_beta", 32.0),
                initial_context_length=config_dict.get("initial_context_length", 4096),
            )
        else:
            # Fallback to GPT-OSS-20B defaults
            return ModelConfig()


class TokenGenerator:
    """Token generator for JAX backend.
    
    Provides a unified interface matching other backends (torch, triton, vllm).
    Automatically detects checkpoint format (Orbax or SafeTensors) and loads weights.
    
    Example:
        >>> generator = TokenGenerator("checkpoint/", max_context_length=4096)
        >>> tokens = [1, 2, 3]  # Prompt tokens
        >>> for token, logprob in generator.generate(tokens, max_tokens=10):
        ...     print(f"Token: {token}, logprob: {logprob}")
    """
    
    def __init__(self, checkpoint_path: str, max_context_length: int = 4096):
        """Initialize token generator.
        
        Args:
            checkpoint_path: Path to checkpoint directory (Orbax or SafeTensors)
            max_context_length: Maximum context length for generation
        """
        self.checkpoint_path = checkpoint_path
        self.max_context_length = max_context_length
        
        # Detect checkpoint format
        self.checkpoint_format = _detect_checkpoint_format(checkpoint_path)
        print(f"[TokenGenerator] Detected checkpoint format: {self.checkpoint_format}")
        
        # Load configuration
        self.config = _load_config_from_checkpoint(checkpoint_path, self.checkpoint_format)
        print(f"[TokenGenerator] Model config: {self.config.num_hidden_layers} layers, "
              f"{self.config.hidden_size} hidden size")
        
        # Load weights
        print(f"[TokenGenerator] Loading weights...")
        if self.checkpoint_format == "orbax":
            loader = OrbaxWeightLoader(checkpoint_path)
            self.params = loader.load_params(show_progress=True)
        else:
            loader = WeightLoader(checkpoint_path)
            self.params = loader.load_params(self.config, show_progress=True)
        
        # Initialize model
        print(f"[TokenGenerator] Initializing model...")
        self.model = Transformer(config=self.config)
        
        # Initialize RNG key for temperature sampling
        self.rng_key = jax.random.PRNGKey(42)
        
        print(f"[TokenGenerator] Ready for generation")
    
    def generate(
        self,
        tokens: List[int],
        stop_tokens: Optional[List[int]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        return_logprobs: bool = False
    ) -> Iterator[Tuple[int, float]]:
        """Generate tokens from prompt.
        
        Args:
            tokens: Input token IDs (prompt)
            stop_tokens: List of token IDs that stop generation (optional)
            temperature: Sampling temperature (0.0 = greedy)
            max_tokens: Maximum number of tokens to generate (None = unlimited)
            return_logprobs: Whether to return log probabilities
            
        Yields:
            Tuple of (token_id, logprob) for each generated token
            If return_logprobs=False, logprob will be 0.0
        """
        if stop_tokens is None:
            stop_tokens = []
        
        # Set max_new_tokens (use a large value, we'll stop early if needed)
        max_new_tokens = max_tokens if max_tokens is not None else 1000
        
        # Prepare RNG key for temperature sampling
        rng_key = self.rng_key if temperature > 0.0 else None
        
        # Use a callback to yield tokens incrementally
        generated_count = [0]  # Use list to allow modification in nested function
        should_stop = [False]  # Flag to stop generation
        
        def token_callback(token: int):
            """Callback called for each generated token."""
            # Check stop tokens
            if token in stop_tokens:
                should_stop[0] = True
                return
            
            # Check max_tokens
            if max_tokens is not None and generated_count[0] >= max_tokens:
                should_stop[0] = True
                return
            
            generated_count[0] += 1
        
        # Generate tokens using inference.generate() with callback
        # We'll need to modify this to support early stopping
        # For now, generate all tokens and filter afterwards
        try:
            generated_tokens = generate(
                model=self.model,
                params=self.params,
                prompt_tokens=tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                rng_key=rng_key,
                show_progress=False,
                token_callback=token_callback,
                use_kv_cache=True,
                config=self.config
            )
        except StopIteration:
            # Early stop via callback (if we implement it)
            pass
        
        # Extract only the newly generated tokens (after prompt)
        new_tokens = generated_tokens[len(tokens):]
        
        # Yield tokens one by one, checking stop conditions
        for i, token in enumerate(new_tokens):
            # Check stop tokens
            if token in stop_tokens:
                break
            
            # Check max_tokens
            if max_tokens is not None and i >= max_tokens:
                break
            
            # For now, logprob is 0.0 (we'd need to modify generate() to return logprobs)
            logprob = 0.0
            yield (token, logprob)
