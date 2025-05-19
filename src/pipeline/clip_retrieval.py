# /home/sysadm/Music/MedXpert/src/pipeline/clip_retrieval.py

from PIL import Image
import os
from src.core.search_engine import search_image_by_text, search_text_by_image

def retrieve_top_k(query, mode="text", k=5):
    """
    Retrieve top k matches for the given query.
    
    Args:
        query: Either a text string or an image path/PIL Image
        mode: Either "text" (for text->image search) or "image" (for image->text search)
        k: Number of results to return
        
    Returns:
        Tuple of (indices, similarity scores)
    """
    if mode == "text":
        if not isinstance(query, str):
            raise ValueError("For text mode, query must be a string")
        return search_image_by_text(query, k=k)
    
    elif mode == "image":
        # Handle different image input formats
        if isinstance(query, str):
            # Check if the path exists
            if not os.path.exists(query):
                raise ValueError(f"Image path does not exist: {query}")
            # No need to load the image here, search_text_by_image will handle it
        elif not isinstance(query, Image.Image):
            raise ValueError("For image mode, query must be either an image path or PIL Image")
        
        return search_text_by_image(query, k=k)
    
    else:
        raise ValueError("mode must be 'text' or 'image'")