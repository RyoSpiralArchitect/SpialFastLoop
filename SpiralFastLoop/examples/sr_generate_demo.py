\
"""
Demo for Surprise→Repair (Surprisal Sandwich) generation using HuggingFace.
Requires: transformers
"""
from spiralfastloop.extras.surprisal_sandwich import surprise_repair_generate

def main():
    txt = surprise_repair_generate(
        prompt="逆説的だが意味は通る短文で、驚きを中盤に入れて最後に回収してください：",
        main_name="Qwen/Qwen2.5-0.5B",
        tiny_name="distilgpt2",
        max_new_tokens=96,
        middle=(0.45, 0.7),
        alpha=10.0, topk=5, mu=0.4
    )
    print(txt)

if __name__ == "__main__":
    main()
