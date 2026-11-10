"""Explicit region and listing corroboration for immutable placement corrections."""
try:
    from . import placement_transform
except ImportError:
    import placement_transform

VERSION = 'placement_from_explicit_region_v2'

def render(root, spec):
    return placement_transform.render(root, spec, explicit_region=True)
