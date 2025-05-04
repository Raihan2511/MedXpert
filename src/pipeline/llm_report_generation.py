def generate_report(blip_captions, retrieved_texts, llm_fn):
    prompt = """
Below are findings extracted from multiple images and related radiology texts.

Image Findings (via BLIP):
"""
    for c in blip_captions:
        prompt += f"- {c}\n"
    prompt += "\nReport Texts (via CLIP):\n"
    for t in retrieved_texts:
        prompt += f"- {t}\n"
    prompt += "\nGenerate a summarized radiology report:"

    return llm_fn(prompt)