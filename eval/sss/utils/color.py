"""
Color utilities for SSS metrics.

Provides color parsing, CIELAB conversion, and color distance functions.
"""

import re
import math
from typing import Tuple, List, Optional

# CSS named colors (subset of common colors)
CSS_COLORS = {
    # Basic colors
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    # Extended colors
    "orange": (255, 165, 0),
    "pink": (255, 192, 203),
    "purple": (128, 0, 128),
    "brown": (165, 42, 42),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    # FigureQA colors
    "aqua": (0, 255, 255),
    "bubblegum": (255, 193, 204),
    "cadet blue": (95, 158, 160),
    "chartreuse": (127, 255, 0),
    "chocolate": (210, 105, 30),
    "coral": (255, 127, 80),
    "crimson": (220, 20, 60),
    "dark blue": (0, 0, 139),
    "dark cyan": (0, 139, 139),
    "dark gold": (170, 136, 0),
    "dark gray": (169, 169, 169),
    "dark green": (0, 100, 0),
    "dark khaki": (189, 183, 107),
    "dark magenta": (139, 0, 139),
    "dark olive": (85, 107, 47),
    "dark orange": (255, 140, 0),
    "dark orchid": (153, 50, 204),
    "dark red": (139, 0, 0),
    "dark salmon": (233, 150, 122),
    "dark seafoam": (143, 188, 143),
    "dark slate": (47, 79, 79),
    "dark violet": (148, 0, 211),
    "deep pink": (255, 20, 147),
    "deep sky blue": (0, 191, 255),
    "dim gray": (105, 105, 105),
    "dodger blue": (30, 144, 255),
    "firebrick": (178, 34, 34),
    "forest green": (34, 139, 34),
    "gold": (255, 215, 0),
    "goldenrod": (218, 165, 32),
    "hot pink": (255, 105, 180),
    "indian red": (205, 92, 92),
    "indigo": (75, 0, 130),
    "khaki": (240, 230, 140),
    "lawn green": (124, 252, 0),
    "light blue": (173, 216, 230),
    "light coral": (240, 128, 128),
    "light gold": (255, 236, 139),
    "light gray": (211, 211, 211),
    "light green": (144, 238, 144),
    "light salmon": (255, 160, 122),
    "light seafoam": (32, 178, 170),
    "light sky blue": (135, 206, 250),
    "lime": (0, 255, 0),
    "lime green": (50, 205, 50),
    "maroon": (128, 0, 0),
    "medium aqua": (102, 205, 170),
    "medium blue": (0, 0, 205),
    "medium orchid": (186, 85, 211),
    "medium purple": (147, 112, 219),
    "medium seafoam": (60, 179, 113),
    "midnight blue": (25, 25, 112),
    "navy blue": (0, 0, 128),
    "olive": (128, 128, 0),
    "olive drab": (107, 142, 35),
    "orchid": (218, 112, 214),
    "pale green": (152, 251, 152),
    "peru": (205, 133, 63),
    "plum": (221, 160, 221),
    "rebecca purple": (102, 51, 153),
    "royal blue": (65, 105, 225),
    "saddle brown": (139, 69, 19),
    "salmon": (250, 128, 114),
    "sandy brown": (244, 164, 96),
    "seafoam": (46, 139, 87),
    "sienna": (160, 82, 45),
    "silver": (192, 192, 192),
    "sky blue": (135, 206, 235),
    "slate": (112, 128, 144),
    "slate blue": (106, 90, 205),
    "steel blue": (70, 130, 180),
    "tan": (210, 180, 140),
    "teal": (0, 128, 128),
    "tomato": (255, 99, 71),
    "turquoise": (64, 224, 208),
    "violet": (238, 130, 238),
    "violet red": (208, 32, 144),
    "web gray": (128, 128, 128),
    "web green": (0, 128, 0),
    "web maroon": (128, 0, 0),
    "web purple": (128, 0, 128),
    "wheat": (245, 222, 179),
    "yellow green": (154, 205, 50),
}


def parse_color(color: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse color string to RGB tuple.

    Supports: hex (#RGB, #RRGGBB), rgb(r,g,b), rgba(r,g,b,a), named colors.

    Returns:
        RGB tuple (0-255) or None if parsing fails.
    """
    if not color:
        return None

    color = str(color).strip()

    # Hex format
    if color.startswith("#"):
        hex_str = color[1:]
        try:
            if len(hex_str) == 3:
                r = int(hex_str[0] * 2, 16)
                g = int(hex_str[1] * 2, 16)
                b = int(hex_str[2] * 2, 16)
            elif len(hex_str) == 6:
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
            elif len(hex_str) == 8:  # RGBA hex
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
            else:
                return None
            return (r, g, b)
        except ValueError:
            return None

    # RGB/RGBA format
    rgb_match = re.match(r"rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", color)
    if rgb_match:
        return (
            int(rgb_match.group(1)),
            int(rgb_match.group(2)),
            int(rgb_match.group(3)),
        )

    # Named color (case-insensitive)
    color_lower = color.lower()
    if color_lower in CSS_COLORS:
        return CSS_COLORS[color_lower]

    return None


def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Convert RGB to CIELAB color space.

    Args:
        rgb: RGB tuple (0-255)

    Returns:
        LAB tuple (L: 0-100, a: -128 to 127, b: -128 to 127)
    """
    # RGB to linear RGB
    def gamma_correct(c: float) -> float:
        c = c / 255.0
        if c > 0.04045:
            return ((c + 0.055) / 1.055) ** 2.4
        return c / 12.92

    r, g, b = gamma_correct(rgb[0]), gamma_correct(rgb[1]), gamma_correct(rgb[2])

    # Linear RGB to XYZ (sRGB D65)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # XYZ to LAB (D65 reference white)
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t: float) -> float:
        if t > 0.008856:
            return t ** (1 / 3)
        return (7.787 * t) + (16 / 116)

    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)

    return (L, a, b_val)


def delta_e(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    """
    Compute CIEDE2000 color difference (simplified).

    Uses CIE76 formula which is simpler but still perceptually meaningful.

    Args:
        lab1: First LAB color
        lab2: Second LAB color

    Returns:
        Color difference (0 = identical, higher = more different)
        Typical JND (just noticeable difference) is around 2.3
    """
    dL = lab2[0] - lab1[0]
    da = lab2[1] - lab1[1]
    db = lab2[2] - lab1[2]

    return math.sqrt(dL**2 + da**2 + db**2)


def color_distance(color1: str, color2: str) -> float:
    """
    Compute perceptual color distance between two color strings.

    Returns:
        Distance in CIELAB space (0 = identical)
        Returns 100.0 if either color cannot be parsed.
    """
    rgb1 = parse_color(color1)
    rgb2 = parse_color(color2)

    if rgb1 is None or rgb2 is None:
        return 100.0  # Max distance for unparseable colors

    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)

    return delta_e(lab1, lab2)


def color_similarity(color1: str, color2: str, scale: float = 50.0) -> float:
    """
    Compute color similarity (0 to 1).

    Args:
        color1: First color string
        color2: Second color string
        scale: Distance at which similarity is exp(-1) ~ 0.37

    Returns:
        Similarity score (1 = identical, 0 = very different)
    """
    dist = color_distance(color1, color2)
    return math.exp(-dist / scale)


def color_emd(colors1: List[str], colors2: List[str]) -> float:
    """
    Compute Earth Mover's Distance between two color arrays.

    Uses a simplified approach: compute pairwise distances and use
    Hungarian matching to find optimal assignment, then average.

    Args:
        colors1: First list of color strings
        colors2: Second list of color strings

    Returns:
        Similarity score (0 to 1), where 1 = identical
    """
    if not colors1 and not colors2:
        return 1.0
    if not colors1 or not colors2:
        return 0.0

    # Parse all colors to LAB
    labs1 = []
    for c in colors1:
        rgb = parse_color(c)
        if rgb:
            labs1.append(rgb_to_lab(rgb))

    labs2 = []
    for c in colors2:
        rgb = parse_color(c)
        if rgb:
            labs2.append(rgb_to_lab(rgb))

    if not labs1 or not labs2:
        return 0.0

    # Build cost matrix
    n1, n2 = len(labs1), len(labs2)
    n = max(n1, n2)

    # Use simple greedy matching for efficiency
    # (Full EMD with scipy would require more complex setup)
    total_dist = 0.0
    used2 = set()

    for lab1 in labs1:
        best_dist = float("inf")
        best_j = -1
        for j, lab2 in enumerate(labs2):
            if j not in used2:
                d = delta_e(lab1, lab2)
                if d < best_dist:
                    best_dist = d
                    best_j = j
        if best_j >= 0:
            used2.add(best_j)
            total_dist += best_dist
        else:
            # No match available, use max penalty
            total_dist += 100.0

    # Add penalty for unmatched colors in colors2
    total_dist += 100.0 * max(0, n2 - n1)

    # Normalize by max(n1, n2) and convert to similarity
    avg_dist = total_dist / n
    return math.exp(-avg_dist / 50.0)
