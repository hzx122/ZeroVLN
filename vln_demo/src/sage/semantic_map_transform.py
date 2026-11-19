"""Cache map bounds while retaining the supplied official coordinate transform."""


def cached_reverse_mapping(original_reverse):
    # Maps remain alive in their evaluators. Keep their identities in this
    # scorer-local cache, rather than importing a separate upstream scorer.
    boundaries = {}

    def reverse(x, y, map_data, flip_x=True, flip_y=True, negate_xy=True):
        key = id(map_data)
        if key not in boundaries:
            coords = [point for item in map_data for point in item.get("mask_coords_m", [])]
            if coords:
                ys, xs = zip(*[(float(y0), float(x0)) for y0, x0 in coords])
                reduced = [{"mask_coords_m": [[min(ys), min(xs)], [max(ys), max(xs)]]}]
            else:
                reduced = []
            boundaries[key] = reduced
        return original_reverse(x, y, boundaries[key], flip_x=flip_x,
                                flip_y=flip_y, negate_xy=negate_xy)

    return reverse
