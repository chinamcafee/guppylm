"""Entry point for: python -m catlm"""

import os
import sys

CHECKPOINT_PATH = "checkpoints_catlm/best_model.pt"
TOKENIZER_PATH = "data_cat_zh/tokenizer.json"


def main():
    if len(sys.argv) < 2:
        print("CatLM — A tiny cat brain")
        print()
        print("Usage:")
        print("  python -m catlm train          Train the model")
        print("  python -m catlm resume-train   Resume training from latest checkpoint")
        print("  python -m catlm prepare        Generate data & train tokenizer")
        print("  python -m catlm chat           Chat with Cat")
        return

    cmd = sys.argv[1]
    sys.argv = sys.argv[1:]

    if cmd == "prepare":
        from .prepare_data import prepare
        prepare()

    elif cmd == "train":
        from .train import train
        train()

    elif cmd == "resume-train":
        from .train import resume_train
        checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None
        resume_train(checkpoint_path)

    elif cmd == "chat":
        if not os.path.exists(CHECKPOINT_PATH):
            print("Model not found. Train your own first:\n")
            print("  python -m catlm prepare")
            print("  python -m catlm train")
            print("  python -m catlm resume-train")
            return

        from .inference import main as inference_main
        inference_main()

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python -m catlm' for usage.")


main()
