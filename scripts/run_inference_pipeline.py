# --- Main pipeline script ---
if __name__ == "__main__":
    import json
    from PIL import Image
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from src.pipeline.clip_retrieval import retrieve_top_k
    from src.pipeline.blip_captioning import generate_blip_captions
    from src.pipeline.llm_report_generation import generate_report
    from src.llm_providers import deepseek_llm  # ✅ USE DEEPSEEK HERE

    # Step 1: Query input
    user_query = "Chest x-ray showing possible effusion"
    indices, _ = retrieve_top_k(user_query, mode="text", k=3)

    # Step 2: Resolve image paths and texts
    with open("data/processed/texts/test.json") as f:
        dataset = json.load(f)
    top_samples = [dataset[i] for i in indices]
    image_paths = [s["image_id"] for s in top_samples]
    texts = [s["text"] for s in top_samples]

    # Step 3: BLIP captions
    blip_outputs = generate_blip_captions(image_paths)

    # Step 4: LLM generation using DeepSeek
    report = generate_report(blip_outputs, texts, deepseek_llm)

    print("\n\nGenerated Report:\n", report)
