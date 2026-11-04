"""Train one fresh single-layer adapter per invocation (1-based layer selection)."""

from train_lora import main

if __name__ == "__main__":
    main(single=True)
