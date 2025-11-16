"""Interactive chat interface for GPT-OSS models.

Supports multiple backends including JAX, PyTorch, Triton, and vLLM.
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    """Main entry point for chat interface."""
    parser = argparse.ArgumentParser(
        description="Interactive chat interface for GPT-OSS models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # JAX backend
  python -m gpt_oss.chat --backend jax weights/gpt-oss-20b

  # With custom temperature and max tokens
  python -m gpt_oss.chat --backend jax weights/gpt-oss-20b --temperature 0.8 --max-tokens 200

  # With streaming output
  python -m gpt_oss.chat --backend jax weights/gpt-oss-20b --stream
        """
    )
    
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint directory"
    )
    
    parser.add_argument(
        "--backend",
        choices=["jax", "triton", "torch", "vllm"],
        default="jax",
        help="Backend to use for inference (default: jax)"
    )
    
    parser.add_argument(
        "--context",
        type=int,
        default=4096,
        help="Maximum context length (default: 4096)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per turn (default: 100)"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output character-by-character"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize backend
    print(f"Initializing {args.backend.upper()} backend...")
    
    match args.backend:
        case "jax":
            # Configure XLA flags for optimal performance
            os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=true --xla_cpu_enable_fast_min_max=true"
            
            # Set compilation cache directory
            cache_dir = Path.home() / ".cache" / "jax"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
            
            # Import JAX TokenGenerator
            from gpt_oss.jax.token_generator import TokenGenerator
            
            print(f"Loading checkpoint from: {checkpoint_path}")
            generator = TokenGenerator(
                checkpoint_path=str(checkpoint_path),
                max_context_length=args.context
            )
            
        case "triton":
            print("Triton backend not yet implemented", file=sys.stderr)
            sys.exit(1)
            
        case "torch":
            print("PyTorch backend not yet implemented", file=sys.stderr)
            sys.exit(1)
            
        case "vllm":
            print("vLLM backend not yet implemented", file=sys.stderr)
            sys.exit(1)
            
        case _:
            print(f"Unknown backend: {args.backend}", file=sys.stderr)
            sys.exit(1)
    
    print("\n" + "="*60)
    print("Chat interface ready! Type 'quit', 'exit', or 'q' to exit.")
    print("="*60 + "\n")
    
    # Chat loop
    try:
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break
            
            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Generate response
            try:
                if args.stream:
                    # Streaming output
                    print("Assistant: ", end="", flush=True)
                    response_chars = generator.generate(
                        prompt=user_input,
                        temperature=args.temperature,
                        max_new_tokens=args.max_tokens,
                        stream=True
                    )
                    for char in response_chars:
                        print(char, end="", flush=True)
                    print()  # Newline after response
                else:
                    # Non-streaming output
                    print("Assistant: ", end="", flush=True)
                    response = generator.generate(
                        prompt=user_input,
                        temperature=args.temperature,
                        max_new_tokens=args.max_tokens,
                        stream=False
                    )
                    print(response)
                    
            except KeyboardInterrupt:
                print("\n\nGeneration interrupted. Type 'quit' to exit or continue chatting.")
                continue
            except Exception as e:
                print(f"\nError during generation: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                continue
            
            print()  # Blank line between turns
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
