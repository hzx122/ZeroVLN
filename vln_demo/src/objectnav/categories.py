"""Preserve Gary's source categories as separate console asset sets."""

SOURCE_CATEGORIES = (
    "hm3d_objectnav_train",
    "mp3d_objectnav_train",
    "hm3d_insight",
    "habitatgs_insight",
    "habitatgs_longrange",
)
BACKEND_CATEGORIES = {
    "gary-" + category.replace("_", "-"): category
    for category in SOURCE_CATEGORIES
}


def select_category(rows, category=None):
    if category is None:
        return rows
    if category not in SOURCE_CATEGORIES:
        raise ValueError("unknown Gary source category: " + category)
    selected = [row for row in rows if row["category"] == category]
    if not selected:
        raise ValueError("empty Gary source category: " + category)
    return selected


def backend_definitions(builder, enabled):
    return {
        key: {
            "label": key,
            "desc": "Gary 原始分类：" + category
                    + "；ObjectNav 指定目标点评分，非官方 category-wide 指标。",
            "goalless": False,
            "hidden": not enabled,
            "build": lambda args, category=category: builder(args, category=category),
        }
        for key, category in BACKEND_CATEGORIES.items()
    }


def expected_counts(rows):
    unknown = {row["category"] for row in rows} - set(SOURCE_CATEGORIES)
    if unknown:
        raise ValueError("unregistered Gary categories: " + ", ".join(sorted(unknown)))
    return {
        key: {"episodes": len(selected), "scenes": len({row["scene"] for row in selected})}
        for key, category in BACKEND_CATEGORIES.items()
        for selected in [select_category(rows, category)]
    }
