"""Validate evaluator scalars without changing navigation geometry or scores."""
import math


def scalar_metrics(metrics):
    values = {key: value.item() if hasattr(value, 'item') else value
              for key, value in metrics.items()
              if isinstance(value, (int, float, bool)) or getattr(value, 'shape', None) == ()}
    invalid = [key for key, value in values.items() if not math.isfinite(value)]
    return {key: None if key in invalid else value for key, value in values.items()}, invalid


def initial_geometry_error(metrics):
    distance = metrics.get('distance_to_goal')
    return distance is not None and not math.isfinite(float(distance))


def finite_result(result):
    return all(isinstance(value, (int, float, bool)) and math.isfinite(value)
               for value in result.get('metrics', {}).values())
