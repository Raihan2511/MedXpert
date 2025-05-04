def retrieve_top_k(query, mode="text", k=5):
    from src.core.search_engine import search_image_by_text, search_text_by_image
    if mode == "text":
        return search_image_by_text(query, k=k)
    elif mode == "image":
        return search_text_by_image(query, k=k)
    else:
        raise ValueError("mode must be 'text' or 'image'")