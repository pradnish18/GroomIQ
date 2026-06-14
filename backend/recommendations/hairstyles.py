STYLE_RULES = {
    ("Oval", "curly"): [
        "Curly Fringe",
        "Low Fade",
        "Textured Crop"
    ],

    ("Round", "straight"): [
        "Pompadour",
        "High Fade",
        "Side Part"
    ]
}

def recommend_styles(face_shape, hair_type):
    return STYLE_RULES.get(
        (face_shape, hair_type.lower()),
        ["Classic Cut"]
    )
"""
GroomIQ Hairstyle Recommendation Engine

Combines:
- Face Shape
- Hair Type
- Hair Health

Returns personalized hairstyle recommendations.
"""

from typing import Dict, List


STYLE_RULES = {
    ("Oval", "straight"): [
        "Pompadour",
        "Quiff",
        "Slick Back",
        "Textured Crop"
    ],

    ("Oval", "wavy"): [
        "Bro Flow",
        "Wavy Quiff",
        "Layered Cut",
        "Textured Waves"
    ],

    ("Oval", "curly"): [
        "Curly Fringe",
        "Low Fade",
        "Textured Crop",
        "Curly Undercut"
    ],

    ("Round", "straight"): [
        "Pompadour",
        "High Fade",
        "Side Part",
        "Brush Up"
    ],

    ("Round", "wavy"): [
        "Wavy Pompadour",
        "High Fade",
        "Messy Top",
        "Side Sweep"
    ],

    ("Round", "curly"): [
        "Curly High Fade",
        "Curly Top",
        "Textured Curls",
        "Curly Faux Hawk"
    ],

    ("Square", "straight"): [
        "Crew Cut",
        "Buzz Cut",
        "French Crop",
        "Ivy League"
    ],

    ("Square", "wavy"): [
        "Wavy Side Part",
        "Classic Taper",
        "Medium Layers",
        "Brush Back"
    ],

    ("Square", "curly"): [
        "Curly Crop",
        "Curly Fade",
        "Short Curls",
        "Textured Top"
    ],

    ("Heart", "straight"): [
        "Side Fringe",
        "Textured Quiff",
        "Classic Taper",
        "Side Sweep"
    ],

    ("Heart", "wavy"): [
        "Messy Waves",
        "Layered Fringe",
        "Wavy Quiff",
        "Medium Flow"
    ],

    ("Heart", "curly"): [
        "Curly Fringe",
        "Loose Curls",
        "Textured Curly Top",
        "Curly Layers"
    ],

    ("Diamond", "straight"): [
        "Classic Taper",
        "Textured Crop",
        "Side Part",
        "Brush Up"
    ],

    ("Diamond", "wavy"): [
        "Wavy Layers",
        "Medium Flow",
        "Messy Top",
        "Wavy Side Part"
    ],

    ("Diamond", "curly"): [
        "Curly Fringe",
        "Curly Crop",
        "Curly Top",
        "Loose Curls"
    ]
}


HEALTH_FILTERS = {
    "dry": ["Avoid excessive heat styling"],
    "frizzy": ["Use anti-frizz products"],
    "hairfall": ["Prefer low-maintenance cuts"],
    "healthy": ["Suitable for most hairstyles"]
}


DEFAULT_STYLES = [
    "Classic Cut",
    "Textured Crop",
    "Side Part"
]


def recommend_styles(face_shape: str, hair_type: str) -> List[str]:
    return STYLE_RULES.get(
        (face_shape, hair_type.lower()),
        DEFAULT_STYLES
    )


def get_style_profile(
    face_shape: str,
    hair_type: str,
    hair_health: str = "healthy"
) -> Dict:

    styles = recommend_styles(face_shape, hair_type)

    return {
        "face_shape": face_shape,
        "hair_type": hair_type,
        "hair_health": hair_health,
        "recommended_styles": styles,
        "health_notes": HEALTH_FILTERS.get(
            hair_health.lower(),
            []
        )
    }