import cv2
import numpy as np

def fast_guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 8, eps: float = 1e-3, s: int = 4) -> np.ndarray:
    """
    Fast Guided Filter to align chromatic channels with original high-resolution ink luminance.
    Prevents color bleeding across panel borders, character lines, and clothes.

    guide: single channel float32 in [0, 1] (e.g. normalized luminance)
    src: single channel float32 (e.g. Lab a or b chroma)
    r: filter radius
    eps: regularization coefficient
    s: subsampling factor for high-speed computation
    """
    h, w = guide.shape[:2]
    small_h, small_w = max(1, h // s), max(1, w // s)
    small_guide = cv2.resize(guide, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_src = cv2.resize(src, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_r = max(1, r // s)

    mean_I = cv2.boxFilter(small_guide, cv2.CV_32F, (small_r, small_r))
    mean_p = cv2.boxFilter(small_src, cv2.CV_32F, (small_r, small_r))
    mean_Ip = cv2.boxFilter(small_guide * small_src, cv2.CV_32F, (small_r, small_r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(small_guide * small_guide, cv2.CV_32F, (small_r, small_r))
    var_I = mean_II - mean_I * mean_I

    a_coeff = cov_Ip / (var_I + eps)
    b_coeff = mean_p - a_coeff * mean_I

    mean_a = cv2.boxFilter(a_coeff, cv2.CV_32F, (small_r, small_r))
    mean_b = cv2.boxFilter(b_coeff, cv2.CV_32F, (small_r, small_r))

    mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)

    return mean_a * guide + mean_b

def create_line_art_mask(gray_img: np.ndarray, threshold: int = 70) -> np.ndarray:
    """
    Extracts deep ink lines from manga page.
    Returns float32 mask in [0, 1] where 1.0 represents solid black ink lines.
    """
    # Ink lines have low luminance
    line_mask = (gray_img < threshold).astype(np.float32)
    # Anti-alias with Gaussian blur for smooth gradient transition
    line_mask = cv2.GaussianBlur(line_mask, (3, 3), 0)
    return line_mask

def detect_page_gutters(gray_img: np.ndarray) -> np.ndarray:
    """
    Detects outer margins and panel gutters to ensure pure clean white page boundaries.
    """
    h, w = gray_img.shape[:2]
    ink = (gray_img < 85).astype(np.uint8) * 255

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    horiz_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, horiz_kernel)
    vert_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vert_kernel)
    panel_borders = cv2.bitwise_or(horiz_lines, vert_lines)
    panel_borders_closed = cv2.dilate(panel_borders, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    all_ink = cv2.bitwise_or(ink, panel_borders_closed)
    white = (gray_img > 215).astype(np.uint8) * 255
    white[all_ink > 0] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    gutter_mask = np.zeros((h, w), dtype=np.float32)

    for i in range(1, num_labels):
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        touches_edge = (x <= 6) or (y <= 6) or (x + bw >= w - 6) or (y + bh >= h - 6)
        if touches_edge and (area > 2500):
            comp_mask = (labels == i)
            if float(np.mean(gray_img[comp_mask])) > 230:
                gutter_mask[comp_mask] = 1.0

    gutter_mask = cv2.dilate(gutter_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return gutter_mask

def create_speech_bubble_mask(gray_img: np.ndarray, white_thresh: int = 246) -> np.ndarray:
    """
    Detects bounded speech bubbles containing dialog text.
    Protects them from picking up colors while NEVER falsely masking out open character line art or white skin.
    Returns float32 mask in [0, 1] where 1.0 indicates protected speech bubble.
    """
    h, w = gray_img.shape[:2]
    ink = (gray_img < 85).astype(np.uint8) * 255
    ink_closed = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    interiors = cv2.bitwise_not(ink_closed)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(interiors)
    bubble_mask = np.zeros_like(gray_img, dtype=np.float32)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Speech bubbles are bounded enclosed regions (typically 1,000 to 45,000 px)
        if area < 1000 or area > 45000:
            continue
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # Skip if it touches the outer canvas borders (gutters / page margins)
        if x <= 2 or y <= 2 or (x + bw) >= (w - 2) or (y + bh) >= (h - 2):
            continue

        comp = (labels == i)
        mean_lum = float(np.mean(gray_img[comp]))
        if mean_lum < float(white_thresh):
            continue

        # Inspect internal dialogue text letters
        patch = gray_img[y:y+bh, x:x+bw]
        text_ratio = np.sum(patch < 80) / float(bw * bh)
        aspect = max(bw, bh) / max(1, min(bw, bh))

        # Real bubbles have dialog lettering inside and compact aspect ratio
        if aspect < 3.2 and text_ratio >= 0.045:
            bubble_mask[comp] = 1.0

    # Soften transition slightly
    bubble_mask = cv2.GaussianBlur(bubble_mask, (5, 5), 0)
    return bubble_mask

def composite_luminance_and_chroma(
    original_bgr: np.ndarray,
    colorized_bgr: np.ndarray,
    color_strength: float = 0.80,
    line_preservation: float = 0.90,
    speech_bubble_protection: bool = True,
    use_guided_filter: bool = True,
    vivid_mode: bool = False
) -> np.ndarray:
    """
    Composites AI-predicted chroma with original high-resolution luminance.
    - Strictly preserves original grayscale shading, hatching, screentones, and contrast.
    - Prevents color bleeding across ink contours.
    - Softly tints bright white character line-art so colors are vibrant instead of washing out into pure white.
    - Keeps true speech bubbles and dialogs clean white.
    """
    orig_lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_lab = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    L_orig = orig_lab[:, :, 0]
    L_pred = color_lab[:, :, 0]
    a_chroma = color_lab[:, :, 1]
    b_chroma = color_lab[:, :, 2]

    # Guide chroma with original luminance to snap colors exactly to line contours
    if use_guided_filter:
        norm_L = L_orig / 255.0
        a_chroma = fast_guided_filter(norm_L, a_chroma, r=8, eps=1e-3, s=4)
        b_chroma = fast_guided_filter(norm_L, b_chroma, r=8, eps=1e-3, s=4)

    # Apply Color Strength (0.0 = pure neutral B&W, 1.0 = full AI saturation)
    # Neutral in Lab is 128.0
    a_blended = 128.0 + (a_chroma - 128.0) * float(color_strength)
    b_blended = 128.0 + (b_chroma - 128.0) * float(color_strength)

    # Speech Bubble & Gutter Margin Protection: clean neutral white for dialogues & gutters
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    bubble_mask = np.zeros_like(L_orig, dtype=np.float32)
    gutter_mask = detect_page_gutters(gray)

    if speech_bubble_protection:
        bubble_mask = create_speech_bubble_mask(gray)
        neutral_mask = np.clip(bubble_mask + gutter_mask, 0.0, 1.0)
        # Invert chroma in speech bubble and margin areas to neutral 128
        a_blended = a_blended * (1.0 - neutral_mask) + 128.0 * neutral_mask
        b_blended = b_blended * (1.0 - neutral_mask) + 128.0 * neutral_mask
    else:
        a_blended = a_blended * (1.0 - gutter_mask) + 128.0 * gutter_mask
        b_blended = b_blended * (1.0 - gutter_mask) + 128.0 * gutter_mask

    # Line Art Protection: desaturate deep ink contours towards neutral 128
    if line_preservation > 0:
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        line_mask = create_line_art_mask(gray)
        line_factor = 1.0 - (line_mask * float(line_preservation))
        a_blended = 128.0 + (a_blended - 128.0) * line_factor
        b_blended = 128.0 + (b_blended - 128.0) * line_factor

    # Vivid Heavy Saturation Boost (if requested by Quality/Vivid mode)
    if vivid_mode:
        a_blended = 128.0 + (a_blended - 128.0) * 1.55
        b_blended = 128.0 + (b_blended - 128.0) * 1.55
        if speech_bubble_protection:
            a_blended = a_blended * (1.0 - bubble_mask) + 128.0 * bubble_mask
            b_blended = b_blended * (1.0 - bubble_mask) + 128.0 * bubble_mask

    # White-Area Luminance Modulation:
    # In pure white manga line-art (L > 200), pure L=255 clamps all colors to white in RGB.
    # When chroma is actively predicted, modulate L slightly so rich skin/clothing tones show warmly!
    chroma_dist = np.sqrt((a_blended - 128.0)**2 + (b_blended - 128.0)**2)
    bright_mask = np.clip((L_orig - 195.0) / 60.0, 0.0, 1.0)
    chroma_active_mask = np.clip(chroma_dist / 6.0, 0.0, 1.0)
    tint_weight = 0.55 if vivid_mode else 0.40
    tint_blend = bright_mask * chroma_active_mask * (1.0 - bubble_mask) * tint_weight

    L_final = L_orig * (1.0 - tint_blend) + L_pred * tint_blend

    # Recombine with exact preserved luminance + modulated white areas
    final_lab = np.zeros_like(orig_lab)
    final_lab[:, :, 0] = np.clip(L_final, 0, 255)
    final_lab[:, :, 1] = np.clip(a_blended, 0, 255)
    final_lab[:, :, 2] = np.clip(b_blended, 0, 255)

    result_bgr = cv2.cvtColor(final_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result_bgr
