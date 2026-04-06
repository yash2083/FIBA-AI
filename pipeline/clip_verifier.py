"""
CLIP Verifier — Zero-shot semantic action verification
=======================================================
Uses OpenAI's CLIP (ViT-B/32) for zero-shot text-image similarity.
Runs entirely on CPU — edge-friendly (~150MB model, ~80ms/frame).

Purpose:
  Given key frames from the pipeline, verify that the visual content
  actually matches the queried action. This gives a semantic confidence
  boost that purely motion-based heuristics cannot provide.

  Example: "picking hotdog"
    Positive prompts: "a hand picking up a hotdog", "grabbing a hotdog"
    Negative prompts: "a hotdog on a table", "an empty plate"
    → CLIP similarity score between frames and prompts

Edge-device notes:
  - Uses ViT-B/32 (smallest CLIP) = ~150MB
  - CPU inference: ~80ms per frame
  - Only runs on 3-5 key frames, not every frame
  - Total overhead: <500ms for key frame verification
"""

import numpy as np
from typing import List, Optional, Tuple

# Lazy-loaded globals
_clip_model = None
_clip_preprocess = None
_clip_tokenize = None
_clip_available = None
_clip_device = None


def _load_clip():
    """Lazy-load CLIP model on first use. Returns True if available."""
    global _clip_model, _clip_preprocess, _clip_tokenize, _clip_available, _clip_device

    if _clip_available is not None:
        return _clip_available

    try:
        import torch
        import clip as clip_module

        _clip_device = "cpu"  # Edge-friendly: CPU only
        _clip_model, _clip_preprocess = clip_module.load("ViT-B/32", device=_clip_device)
        _clip_tokenize = clip_module.tokenize
        _clip_model.eval()
        _clip_available = True
        print("[CLIP] ViT-B/32 loaded on CPU — zero-shot verification enabled")
    except ImportError:
        # Try open_clip as fallback
        try:
            import torch
            import open_clip

            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            _clip_tokenize = open_clip.get_tokenizer("ViT-B-32")
            _clip_device = "cpu"
            _clip_model = _clip_model.to(_clip_device)
            _clip_model.eval()
            _clip_available = True
            print("[CLIP] open_clip ViT-B/32 loaded — zero-shot verification enabled")
        except ImportError:
            _clip_available = False
            print("[CLIP] Not available (install: pip install openai-clip torch). "
                  "Falling back to motion-only inference.")

    return _clip_available


def _build_prompts(action_verb: str, object_noun: str, action_category: str) -> Tuple[List[str], List[str]]:
    """Build positive and negative text prompts for CLIP comparison."""
    verb = action_verb.lower()
    obj = object_noun.lower()
    cat = action_category.upper()

    # --- Positive prompts (what we expect to see) ---
    positive = [
        f"a person {verb} a {obj}",
        f"a hand {verb} a {obj}",
        f"someone {verb} the {obj}",
        f"{verb} a {obj}",
    ]

    # Category-specific positive prompts
    if cat == "PICK":
        positive += [
            f"a hand grabbing a {obj}",
            f"a hand holding a {obj}",
            f"picking up a {obj}",
            f"a {obj} being held in hand",
        ]
    elif cat == "CUT":
        positive += [
            f"cutting a {obj} with a knife",
            f"a knife slicing through a {obj}",
            f"chopping a {obj}",
        ]
    elif cat == "POUR":
        positive += [
            f"pouring from a {obj}",
            f"liquid being poured",
            f"tilting a {obj} to pour",
        ]
    elif cat == "OPEN":
        positive += [
            f"opening a {obj}",
            f"a {obj} being opened",
            f"unscrewing a {obj}",
        ]
    elif cat == "PLACE":
        positive += [
            f"putting down a {obj}",
            f"placing a {obj} on a surface",
            f"setting a {obj} on a table",
        ]
    elif cat == "MIX":
        positive += [
            f"stirring with a spoon",
            f"mixing ingredients",
            f"whisking in a bowl",
        ]

    # --- Negative prompts (what we don't expect) ---
    negative = [
        f"a {obj} sitting on a table",
        f"an empty kitchen counter",
        "a blank wall",
        "no hands visible",
        f"a {obj} untouched",
    ]

    return positive, negative


def compute_clip_score(
    frames: List[np.ndarray],
    action_verb: str,
    object_noun: str,
    action_category: str,
) -> Tuple[float, str]:
    """
    Compute CLIP-based semantic action verification score.

    Args:
        frames:          List of BGR numpy frames (key frames)
        action_verb:     e.g. "picking"
        object_noun:     e.g. "hot dog"
        action_category: e.g. "PICK"

    Returns:
        (score, explanation):
            score: 0.0–1.0 semantic confidence
            explanation: human-readable text
    """
    if not _load_clip():
        return 0.5, "[CLIP unavailable — using motion-only score]"

    if not frames:
        return 0.0, "[No frames to verify]"

    import torch
    from PIL import Image

    positive_prompts, negative_prompts = _build_prompts(action_verb, object_noun, action_category)
    all_prompts = positive_prompts + negative_prompts
    n_pos = len(positive_prompts)

    # Tokenize all prompts
    if callable(_clip_tokenize):
        text_tokens = _clip_tokenize(all_prompts)
        if hasattr(text_tokens, 'to'):
            text_tokens = text_tokens.to(_clip_device)
    else:
        text_tokens = _clip_tokenize(all_prompts).to(_clip_device)

    # Process frames
    frame_scores = []
    with torch.no_grad():
        # Encode text features once
        text_features = _clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for frame in frames[:5]:  # Max 5 frames for speed
            try:
                # Convert BGR to RGB PIL Image
                rgb = frame[:, :, ::-1]  # BGR -> RGB
                pil_img = Image.fromarray(rgb)
                img_tensor = _clip_preprocess(pil_img).unsqueeze(0).to(_clip_device)

                # Encode image
                image_features = _clip_model.encode_image(img_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Compute similarities
                similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()

                # Positive vs negative scoring
                pos_sim = float(np.mean(similarities[:n_pos]))
                neg_sim = float(np.mean(similarities[n_pos:]))

                # Score: how much more positive than negative
                # CLIP similarities are typically in [0.15, 0.35] range
                margin = pos_sim - neg_sim
                frame_score = float(np.clip((margin + 0.05) / 0.15, 0.0, 1.0))
                frame_scores.append(frame_score)
            except Exception as e:
                print(f"[CLIP] Frame processing error: {e}")
                frame_scores.append(0.5)

    if not frame_scores:
        return 0.5, "[CLIP processing failed]"

    # Aggregate: use max of top-2 scores (best frames matter most)
    sorted_scores = sorted(frame_scores, reverse=True)
    top_scores = sorted_scores[:min(2, len(sorted_scores))]
    final_score = float(np.mean(top_scores))

    explanation = (
        f"[CLIP ViT-B/32] Semantic verification: {final_score:.2f} "
        f"(best frames: {', '.join(f'{s:.2f}' for s in top_scores)}). "
        f"Matched '{action_verb} {object_noun}' against {len(positive_prompts)} positive "
        f"and {len(negative_prompts)} negative prompts."
    )

    return final_score, explanation


def is_available() -> bool:
    """Check if CLIP is available without loading it."""
    try:
        import torch
        try:
            import clip
            return True
        except ImportError:
            pass
        try:
            import open_clip
            return True
        except ImportError:
            pass
    except ImportError:
        pass
    return False


if __name__ == "__main__":
    print("=== CLIP Verifier Test ===")
    print(f"CLIP available: {is_available()}")

    if is_available():
        # Test with a blank frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        score, explanation = compute_clip_score(
            [test_frame], "picking", "hot dog", "PICK"
        )
        print(f"Score: {score:.3f}")
        print(f"Explanation: {explanation}")
    else:
        print("Install CLIP: pip install openai-clip torch torchvision")
        print("Or: pip install open-clip-torch torch torchvision")
