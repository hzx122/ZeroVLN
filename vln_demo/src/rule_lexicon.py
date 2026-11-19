"""Exact navigation aliases shared by the rule executor and NLP front end.

Labels are deliberately not collapsed by substring (a basin cabinet is not a basin).
"""
import re

ALIASES = {
    'basin': ('basin', 'sink', 'washbasin'),
    'sofa': ('sofa', 'couch'), 'bed': ('bed',), 'chair': ('chair',),
    'armchair': ('armchair',), 'table': ('table',), 'desk': ('desk',),
    'coffee table': ('coffee table', 'tea table', 'teatable'),
    'conference table': ('conference table',),
    'bookshelf': ('bookshelf', 'bookcase'), 'cabinet': ('cabinet', 'cupboard'),
    'wardrobe': ('wardrobe',), 'television': ('television', 'tv'),
    'screen': ('screen', 'projection screen'), 'refrigerator': ('refrigerator', 'fridge'),
    'mirror': ('mirror', 'mirrors'), 'bathtub': ('bathtub', 'tub'), 'toilet': ('toilet',),
    'window': ('window',), 'curtain': ('curtain', 'curtains'),
    'doorway': ('doorway', 'opening'),
    'dining table': ('dining table',), 'bedside table': ('bedside table', 'nightstand'),
    'floor lamp': ('floor lamp',), 'ceiling lamp': ('ceiling lamp',),
    'plant': ('plant', 'potted plant'), 'bookshelf unit': ('bookshelf unit', 'shelving unit'),
    'computer': ('computer', 'desktop computer', 'notebook computer', 'laptop'),
    'computer screen': ('computer screen', 'monitor'),
    'keyboard': ('keyboard',), 'tray': ('tray',), 'stool': ('stool','stools'),
    'shower head': ('shower head',), 'table lamp': ('table lamp',),
    'book': ('book','books'), 'pillow': ('pillow','pillows'),
    'blanket': ('blanket','blankets'), 'vase': ('vase',),
    'flowers': ('flowers','flower'), 'towel': ('towel',),
    'shelf': ('shelf',), 'storage rack': ('storage rack',),
}
PLURAL_TARGETS={'mirrors','stools','books','pillows','blankets'}
NAMES = {alias: family for family, aliases in ALIASES.items() for alias in aliases}
NOUN = re.compile(r'\b(' + '|'.join(re.escape(x) for x in sorted(NAMES, key=len, reverse=True)) + r')\b')

ROOMS = {
    'bedroom': 'sleeping area', 'bathroom': 'bathroom area',
    'living room': 'living area', 'living area': 'living area',
    'dining room': 'dining area', 'dining area': 'dining area',
    'kitchen': 'kitchen area', 'study room': 'study area',
    'study': 'study area', 'study area': 'study area',
    'common area': 'common area',
    'living room area': 'living area', 'bathroom area': 'bathroom area',
    'bedroom area': 'sleeping area',
}
ROOM_PATTERN = '(?:'+'|'.join(sorted(map(re.escape, ROOMS), key=len, reverse=True))+')'

