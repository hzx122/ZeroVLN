"""SAGE-3D navigation over InteriorGS rendering and native USD/Recast geometry.

The default physical backend preserves task-designated objects, follows 3-D
support through movement/collision/camera/reward/scoring, and rejects missing
or ambiguous geometry. GVLN trajectory z=0.5 is a placeholder, never evidence
of a floor. SAGE_GEOMETRY_MODE=legacy_xy is only for explicit historical replay.
"""
