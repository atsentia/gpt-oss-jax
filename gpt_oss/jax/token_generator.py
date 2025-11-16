"""TokenGenerator for JAX backend.

Provides a unified interface for token generation that matches other backends.
"""

import jax
import jax.numpy as jnp
from pathlib import Path
from typing import List, Optional, Iterator, Tuple, Union

from .model import Transformer
from .config import ModelConfig
from .inference import generate
from .loader_orbax import OrbaxWeightLoader, load_config_from_orbax
from .loader_safetensors import WeightLoader
from gpt_oss.tokenizer import get_tokenizer


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
        
        # Initialize tokenizer
        self.tokenizer = get_tokenizer()
        
        # Conversation history for chat interface
        self.conversation_history = []
        
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
    
    def _format_conversation(self, include_user_message: Optional[str] = None) -> str:
        """Format conversation history, optionally including a new user message.
        
        Args:
            include_user_message: Optional new user message to include (not added to history)
            
        Returns:
            Formatted conversation string ready for tokenization
        """
        # Format conversation (simple format: "User: ... Assistant: ... User: ...")
        formatted = []
        for msg in self.conversation_history:
            if msg["role"] == "user":
                formatted.append(f"User: {msg['content']}")
            else:
                formatted.append(f"Assistant: {msg['content']}")
        
        # Add new user message if provided (not yet in history)
        if include_user_message is not None:
            formatted.append(f"User: {include_user_message}")
        
        # Add "Assistant: " prefix for next response
        formatted.append("Assistant: ")
        
        return "\n".join(formatted)
    
    def _truncate_conversation(self, max_tokens: int, new_user_message: str):
        """Truncate conversation history if it exceeds context length.
        
        Args:
            max_tokens: Maximum number of tokens allowed
            new_user_message: New user message to include in token count
        """
        # Estimate tokens in current history + new message
        formatted = self._format_conversation(include_user_message=new_user_message)
        # Remove the "Assistant: " suffix we added
        formatted = formatted.rsplit("Assistant: ", 1)[0]
        current_tokens = self.tokenizer.encode(formatted)
        
        # If within limit, no truncation needed
        if len(current_tokens) <= max_tokens:
            return
        
        # Remove oldest messages until we're within limit
        # Keep at least the last user message
        while len(self.conversation_history) > 1:
            # Remove oldest message pair (user + assistant)
            if len(self.conversation_history) >= 2:
                self.conversation_history.pop(0)  # Remove oldest user
                if self.conversation_history and self.conversation_history[0]["role"] == "assistant":
                    self.conversation_history.pop(0)  # Remove corresponding assistant
            
            # Check if we're within limit now
            formatted = self._format_conversation(include_user_message=new_user_message)
            formatted = formatted.rsplit("Assistant: ", 1)[0]
            current_tokens = self.tokenizer.encode(formatted)
            if len(current_tokens) <= max_tokens:
                break
    
    def generate(
        self,
        prompt: Optional[str] = None,
        tokens: Optional[List[int]] = None,
        stop_tokens: Optional[List[int]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        return_logprobs: bool = False,
        stream: bool = False
    ) -> Union[str, Iterator[str], Iterator[Tuple[int, float]]]:
        """Generate response from prompt string or tokens.
        
        Args:
            prompt: Input text (if provided, tokens is ignored)
            tokens: Input token IDs (if prompt not provided)
            stop_tokens: List of token IDs that stop generation (optional)
            temperature: Sampling temperature (0.0 = greedy)
            max_tokens: Maximum number of tokens to generate (deprecated, use max_new_tokens)
            max_new_tokens: Maximum number of tokens to generate (None = unlimited)
            return_logprobs: Whether to return log probabilities (only for token-based generation)
            stream: Whether to stream output character-by-character (only for prompt-based generation)
            
        Returns:
            If prompt provided and stream=False: Generated text (str)
            If prompt provided and stream=True: Iterator[str] yielding characters
            If tokens provided: Iterator[Tuple[int, float]] yielding (token_id, logprob)
        """
        # Handle max_tokens deprecation
        if max_tokens is not None and max_new_tokens is None:
            max_new_tokens = max_tokens
        
        # Determine input tokens
        if prompt is not None:
            # Truncate conversation if needed (before adding user message to history)
            self._truncate_conversation(
                self.max_context_length - (max_new_tokens or 100),
                new_user_message=prompt
            )
            
            # Format conversation with user message (not yet in history)
            formatted_prompt = self._format_conversation(include_user_message=prompt)
            input_tokens = self.tokenizer.encode(formatted_prompt)
            
            # Now add user message to history
            self.conversation_history.append({"role": "user", "content": prompt})
        elif tokens is not None:
            input_tokens = tokens
        else:
            raise ValueError("Either 'prompt' or 'tokens' must be provided")
        
        if stop_tokens is None:
            stop_tokens = []
        
        # Set max_new_tokens (use a large value, we'll stop early if needed)
        max_new_tokens = max_new_tokens if max_new_tokens is not None else 1000
        
        # Prepare RNG key for temperature sampling
        rng_key = self.rng_key if temperature > 0.0 else None
        
        # Collect generated tokens for text-based generation
        generated_tokens_list = []
        
        def token_callback(token: int):
            """Callback called for each generated token."""
            generated_tokens_list.append(token)
        
        # Generate tokens using inference.generate()
        try:
            generated_tokens = generate(
                model=self.model,
                params=self.params,
                prompt_tokens=input_tokens,
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
        new_tokens = generated_tokens[len(input_tokens):]
        
        # Handle stop tokens
        filtered_tokens = []
        for token in new_tokens:
            if token in stop_tokens:
                break
            filtered_tokens.append(token)
        
        # Limit by max_new_tokens
        if max_new_tokens is not None:
            filtered_tokens = filtered_tokens[:max_new_tokens]
        
        # If prompt was provided, return text
        if prompt is not None:
            # Decode generated tokens
            generated_text = self.tokenizer.decode(filtered_tokens)
            
            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": generated_text})
            
            if stream:
                # Stream character-by-character
                def stream_generator():
                    for char in generated_text:
                        yield char
                return stream_generator()
            else:
                return generated_text
        else:
            # Token-based generation: yield (token, logprob) tuples
            def token_generator():
                for token in filtered_tokens:
                    logprob = 0.0  # We'd need to modify generate() to return logprobs
                    yield (token, logprob)
            return token_generator()
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
