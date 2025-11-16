#!/bin/bash
# Harmony format example script for GPT-OSS-20B with JAX backend
# Demonstrates proper Harmony tokenizer usage

set -e

# Default values
CHECKPOINT_PATH="${1:-weights/gpt-oss-20b}"
PROMPT="${2:-What is the capital of France?}"
MAX_TOKENS="${3:-50}"
TEMPERATURE="${4:-0.0}"

echo "============================================================"
echo "Harmony Format Example - GPT-OSS-20B with JAX"
echo "============================================================"
echo ""
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Prompt: $PROMPT"
echo "Max tokens: $MAX_TOKENS"
echo "Temperature: $TEMPERATURE"
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path does not exist: $CHECKPOINT_PATH"
    echo ""
    echo "Usage: $0 [checkpoint_path] [prompt] [max_tokens] [temperature]"
    echo ""
    echo "Example:"
    echo "  $0 weights/gpt-oss-20b \"What is the capital of France?\" 50 0.0"
    exit 1
fi

# Run Python script with Harmony example
python3 << PYTHON_SCRIPT
import jax
import os
from pathlib import Path

# Configure XLA flags for optimal performance
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=true --xla_cpu_enable_fast_min_max=true"

# Set compilation cache directory
cache_dir = Path.home() / ".cache" / "jax"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*Sharding info not provided.*")
warnings.filterwarnings("ignore", category=UserWarning, module="orbax.*")

print("Loading checkpoint and initializing model...")
print("")

# Import modules
from gpt_oss.jax.inference import generate
from gpt_oss.jax.token_generator import TokenGenerator
from gpt_oss.tokenizer import get_tokenizer

# Initialize generator
generator = TokenGenerator("$CHECKPOINT_PATH", max_context_length=4096)

# Get tokenizer for decoding
tokenizer = get_tokenizer()

# User message
user_message = "$PROMPT"

# ANSI color codes for high-contrast terminal output
class Colors:
    # Bold colors for good contrast
    CYAN = '\033[96m'       # Bright cyan for original prompt
    BLUE = '\033[94m'       # Blue for Harmony prompt
    YELLOW = '\033[93m'     # Yellow for full generated response
    GREEN = '\033[92m'      # Green for reasoning
    MAGENTA = '\033[95m'    # Magenta for final answer
    RED = '\033[91m'        # Red for errors/warnings
    BOLD = '\033[1m'        # Bold
    UNDERLINE = '\033[4m'   # Underline
    RESET = '\033[0m'       # Reset to default

print("=" * 60)
print(f"{Colors.BOLD}Harmony Format Example{Colors.RESET}")
print("=" * 60)
print(f"{Colors.CYAN}{Colors.BOLD}a) Original User Message:{Colors.RESET}")
print(f"{Colors.CYAN}   '{user_message}'{Colors.RESET}")
print("")

# Try to use openai-harmony for proper Harmony formatting
try:
    from openai_harmony import (
        load_harmony_encoding,
        HarmonyEncodingName,
        Conversation,
        Message,
        Role
    )
    
    # Load Harmony encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    
    # Create Harmony conversation
    conversation = Conversation.from_messages([
        Message.from_role_and_content(Role.USER, user_message)
    ])
    
    # Render conversation for completion (adds Harmony special tokens)
    prompt_tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    
    # Get Harmony stop tokens (so generation stops at proper boundaries)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    # Decode prompt to show Harmony formatting
    prompt_text = tokenizer.decode(prompt_tokens)

    print(f"{Colors.BLUE}{Colors.BOLD}b) Harmony-Formatted Prompt:{Colors.RESET}")
    print(f"{Colors.BLUE}{prompt_text}{Colors.RESET}")
    print("")
    print("✓ Using openai-harmony API for Harmony formatting")
    print(f"  Prompt tokens: {len(prompt_tokens)}")
    print(f"  Special tokens used: {[t for t in prompt_tokens if t >= 199998]}")
    print(f"  Stop tokens: {stop_token_ids[:10]}... (will stop at Harmony boundaries)")
    print("")
    
    # Generate using raw tokens (bypass TokenGenerator's formatting)
    print("Generating response...")
    print("")
    
    output_tokens, stats = generate(
        model=generator.model,
        params=generator.params,
        prompt_tokens=prompt_tokens,
        max_new_tokens=$MAX_TOKENS,
        temperature=$TEMPERATURE,
        rng_key=jax.random.PRNGKey(42),
        config=generator.config,
        use_kv_cache=True,
        show_progress=False,
        return_stats=True
    )
    
    # Filter out stop tokens manually if needed
    # (generate() doesn't support stop_tokens parameter, so we'll filter after generation)
    filtered_tokens = []
    for token in output_tokens[len(prompt_tokens):]:
        if token in stop_token_ids:
            break
        filtered_tokens.append(token)
    
    # Reconstruct full output with filtered tokens
    output_tokens = prompt_tokens + filtered_tokens
    
    # Decode and display
    output_text = tokenizer.decode(output_tokens)

    # Extract just the generated part (without the prompt)
    generated_only = tokenizer.decode(output_tokens[len(prompt_tokens):])

    print("=" * 60)
    print(f"{Colors.YELLOW}{Colors.BOLD}c) Full Generated Response (raw Harmony format):{Colors.RESET}")
    print("=" * 60)
    print(f"{Colors.YELLOW}{generated_only}{Colors.RESET}")
    print("=" * 60)
    
    # Parse Harmony output to separate reasoning from final answer
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}d) Parsed Harmony Output (extracted channels):{Colors.RESET}")
    print("=" * 60)
    
    try:
        # Parse completion tokens into structured messages
        completion_tokens = output_tokens[len(prompt_tokens):]  # Only the generated tokens
        messages = encoding.parse_messages_from_completion_tokens(completion_tokens, Role.ASSISTANT)
        
        # Extract reasoning and final answer from parsed messages
        reasoning_parts = []
        answer_parts = []
        
        for msg in messages:
            # Check message structure - Harmony messages have recipient/channel info
            msg_dict = msg.to_dict() if hasattr(msg, 'to_dict') else {}
            
            # Look for channel information
            recipient = getattr(msg, 'recipient', None) or msg_dict.get('recipient', '')
            content = getattr(msg, 'content', None) or msg_dict.get('content', '')
            
            if 'analysis' in str(recipient).lower() or 'analysis' in str(content).lower():
                reasoning_parts.append(str(content))
            elif 'main' in str(recipient).lower() or (not recipient and content):
                answer_parts.append(str(content))
            else:
                # Try to extract from content string
                content_str = str(content)
                if '<|channel|>analysis' in content_str:
                    reasoning_parts.append(content_str)
                elif '<|channel|>main' in content_str:
                    answer_parts.append(content_str)
        
        # Display parsed output
        if reasoning_parts:
            print(f"\n{Colors.GREEN}{Colors.BOLD}📊 Reasoning/Analysis (analysis channel):{Colors.RESET}")
            print(f"{Colors.GREEN}" + "-" * 60)
            for part in reasoning_parts:
                print(part)
            print(f"{Colors.RESET}")

        if answer_parts:
            print(f"\n{Colors.MAGENTA}{Colors.BOLD}💬 Final Answer (final channel):{Colors.RESET}")
            print(f"{Colors.MAGENTA}" + "-" * 60)
            for part in answer_parts:
                print(part)
            print(f"{Colors.RESET}")
        
        if not reasoning_parts and not answer_parts:
            # Fallback: show all messages
            print("\n📝 All Messages:")
            for i, msg in enumerate(messages):
                print(f"\nMessage {i+1}:")
                print(f"  {msg}")
        
    except Exception as e:
        # If parsing fails, try manual parsing of Harmony tokens
        print(f"\n{Colors.RED}⚠️  Harmony parsing failed ({e}), trying manual parsing...{Colors.RESET}")

        # Manual parsing: look for <|channel|>analysis<|message|> and <|channel|>main<|message|>
        import re

        # Extract reasoning (analysis channel)
        reasoning_match = re.search(r'<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)', generated_only, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            print(f"\n{Colors.GREEN}{Colors.BOLD}📊 Reasoning/Analysis (analysis channel):{Colors.RESET}")
            print(f"{Colors.GREEN}" + "-" * 60)
            print(reasoning)
            print(f"{Colors.RESET}")

        # Extract final answer (main/final channel)
        answer_match = re.search(r'<\|channel\|>(main|final)<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)', generated_only, re.DOTALL)
        if answer_match:
            answer = answer_match.group(2).strip()
            print(f"\n{Colors.MAGENTA}{Colors.BOLD}💬 Final Answer ({answer_match.group(1)} channel):{Colors.RESET}")
            print(f"{Colors.MAGENTA}" + "-" * 60)
            print(answer)
            print(f"{Colors.RESET}")
        
        if not reasoning_match and not answer_match:
            print(f"\n{Colors.RED}⚠️  Could not parse Harmony channels{Colors.RESET}")
            print(f"{Colors.RED}   Raw output shown above{Colors.RESET}")
    
    print("=" * 60)
    
    print(f"\nTime: {stats['total_time']:.2f}s | Speed: {stats['tokens_per_second']:.2f} tok/s")
    
except ImportError:
    print("⚠️  openai-harmony not available")
    print("   Install with: pip install openai-harmony")
    print("   Falling back to basic tokenizer (may not work correctly)")
    print("")
    
    # Fallback: use TokenGenerator (but it uses plain format, not Harmony)
    response = generator.generate(
        prompt=user_message,
        temperature=$TEMPERATURE,
        max_new_tokens=$MAX_TOKENS,
        stream=False
    )
    
    print("=" * 60)
    print("Generated Response (may be incorrect without Harmony format):")
    print("=" * 60)
    print(response)
    print("=" * 60)
PYTHON_SCRIPT

echo ""
echo "Done!"
