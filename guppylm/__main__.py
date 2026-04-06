"""Entry point for: python -m guppyllm"""

import sys


def main():
    if len(sys.argv) < 2:
        print("GuppyLM — A tiny fish brain")
        print()
        print("Usage:")
        print("  python -m guppyllm train       Train the model")
        print("  python -m guppyllm prepare      Generate data & train tokenizer")
        print("  python -m guppyllm chat         Chat with Guppy")
        return

    cmd = sys.argv[1]
    sys.argv = sys.argv[1:]

    if cmd == "prepare":
        from .prepare_data import prepare
        prepare()

    elif cmd == "train":
        from .train import train
        train()

    elif cmd == "chat":
        from .inference import main
        main()

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python -m guppyllm' for usage.")


main()
