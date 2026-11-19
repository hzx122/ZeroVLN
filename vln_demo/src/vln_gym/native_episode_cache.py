"""Reuse native dataset parsers; retain immutable, content-bound episode bytes.

Each environment gets a private dataset, iterator and freshly deserialized
current episode. No simulator, pathfinder, goal or task state is shared. The
entire original dataset remains addressable; this does not sample or thin it.
"""
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
from threading import RLock
from types import MethodType

_cache = OrderedDict()
_lock = RLock()
_cache_bytes = 0
MAX_ENTRIES = 4
MAX_BYTES = 512 * 1024 * 1024


def owned_text(pool, value):
    """Keep catalog identities outside the native parser's allocation arenas."""
    text = str(value)
    if text not in pool:
        copy = text.encode('utf-8', 'surrogatepass').decode('utf-8', 'surrogatepass')
        pool[copy] = copy
    return pool[text]


def episode_catalog(episodes):
    """Original full scene/episode catalog, without retaining parser strings."""
    identities = (episodes._identities if isinstance(episodes, FrozenEpisodes) else
        ((getattr(ep, 'scene', None) or ep.scene_id.split('/')[-2], ep.episode_id)
         for ep in episodes))
    result, strings = {}, {}
    for scene, episode_id in identities:
        result.setdefault(owned_text(strings, scene), []).append(owned_text(strings, episode_id))
    return result


def enabled():
    value = os.environ.get('VLN_NATIVE_EPISODE_CACHE', '0')
    if value not in ('0', '1'):
        raise ValueError('VLN_NATIVE_EPISODE_CACHE must be 0 or 1')
    return value == '1'


@dataclass(frozen=True)
class EpisodeReference:
    index: int
    scene_id: str


@dataclass(frozen=True)
class EpisodePayloads:
    """Keep native pickles in one allocation, away from the parser's heap.

    Thousands of small retained pickle allocations can pin pages otherwise
    occupied by temporary native episode/goal/path objects after parsing.
    Views deserialize the exact same pickle bytes without copying the blob.
    """
    blob: bytes
    offsets: tuple

    def __len__(self):
        return len(self.offsets)-1

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return memoryview(self.blob)[self.offsets[index]:self.offsets[index+1]]


class FrozenEpisodes(Sequence):
    """Only immutable bytes/references are shared; reads yield private objects."""
    def __init__(self, payloads, identities, references):
        self._payloads = payloads
        self._identities = identities
        self.references = references

    def __len__(self):
        return len(self._payloads)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return pickle.loads(self._payloads[index])

    def find(self, scene, episode_id, *, fallback=False):
        for index, (sc, ep) in enumerate(self._identities):
            if (sc == scene or sc.endswith('-' + scene)) and ep == str(episode_id):
                return self[index]
        if fallback:
            for index, (sc, _) in enumerate(self._identities):
                if sc == scene or sc.endswith('-' + scene):
                    return self[index]
            return self[0] if self else None
        return None


def _bindings(sources):
    return tuple((str(Path(path).resolve()), hashlib.sha256(
        Path(path).read_bytes()).hexdigest()) for path in sources)


def cached_episodes(signature, sources, loader):
    """loader returns (metadata, original typed episodes), without a new parser."""
    global _cache_bytes
    bindings = _bindings(sources)
    key = (json.dumps(signature, sort_keys=True, ensure_ascii=False), bindings)
    with _lock:
        hit = key in _cache
        if hit:
            stored = _cache[key]
            _cache.move_to_end(key)
        else:
            metadata, episodes = loader()
            archive, offsets = io.BytesIO(), [0]
            identities, references = [], []
            owned_strings = {}
            for index, episode in enumerate(episodes):
                archive.write(pickle.dumps(episode, protocol=5))
                offsets.append(archive.tell())
                scene = getattr(episode, 'scene', None) or episode.scene_id.split('/')[-2]
                identities.append((owned_text(owned_strings, scene), owned_text(owned_strings, episode.episode_id)))
                references.append(EpisodeReference(index, owned_text(owned_strings, episode.scene_id)))
            metadata = pickle.dumps(metadata, protocol=5)
            payloads = EpisodePayloads(archive.getvalue(), tuple(offsets))
            archive.close()
            stored = (metadata, payloads, tuple(identities), tuple(references))
            if _bindings(sources) != bindings:
                raise ValueError('native episode source changed while loading')
            size = len(metadata) + len(payloads.blob) + 512 * len(payloads)
            if size <= MAX_BYTES:
                while _cache and (len(_cache) >= MAX_ENTRIES or _cache_bytes + size > MAX_BYTES):
                    _, (_, removed) = _cache.popitem(last=False)
                    _cache_bytes -= removed
                _cache[key] = (stored, size)
                _cache_bytes += size
        if hit:
            stored = stored[0]
    metadata, payloads, identities, references = stored
    return (pickle.loads(metadata), FrozenEpisodes(payloads, identities, references),
            dict(cache_hit=hit, sources=dict(bindings), episodes=len(payloads)))


def private_episode_iterator(dataset, *args, **kwargs):
    # Preserve Habitat's original shuffle/grouping/cycle/repetition behavior,
    # including its isinstance(EpisodeIterator) step_taken contract. Sorting
    # references cannot materialize thousands of episode/goal/path objects.
    from habitat.core.dataset import EpisodeIterator

    class PrivateEpisodeIterator(EpisodeIterator):
        def __next__(self):
            reference = super().__next__()
            return dataset.episodes[reference.index]

    return PrivateEpisodeIterator(dataset.episodes.references, *args, **kwargs)


def cached_habitat_env(evaluator):
    """Original Habitat Env(dataset=...), with the original language mask."""
    from habitat import Env
    dataset, evidence = cached_habitat_dataset(evaluator)
    env = Env(config=evaluator.config, dataset=dataset)
    env._native_episode_cache_evidence = evidence
    return env


def cached_habitat_dataset(evaluator):
    """Build the full private dataset view without creating a simulator."""
    from habitat import make_dataset
    from omegaconf import OmegaConf
    cfg = evaluator.config.habitat.dataset
    language = os.environ.get('RYNNVLN_LANGUAGE', 'en').strip()
    signature = dict(kind='habitat', config=OmegaConf.to_container(cfg, resolve=True),
                     language=language, scenes_dir=str(Path(cfg.scenes_dir).resolve()))
    source = cfg.data_path.format(split=cfg.split)

    def load():
        dataset = make_dataset(id_dataset=cfg.type, config=cfg)
        episodes = dataset.episodes
        if language.lower() != 'all':
            wanted = {x.strip().lower() for x in language.split(',') if x.strip()}
            def keep(ep):
                text = getattr(getattr(ep, 'instruction', None), 'instruction_text', '') or ''
                lang = ('te' if re.search(r'[\u0C00-\u0C7F]', text) else
                        'hi' if re.search(r'[\u0900-\u097F]', text) else
                        'en' if re.search(r'[A-Za-z]', text) else 'other')
                return lang in wanted
            episodes = [ep for ep in episodes if keep(ep)]
        metadata = dict(dataset_class=type(dataset), attributes={
            k: v for k, v in vars(dataset).items() if k != 'episodes'})
        return metadata, episodes

    metadata, episodes, evidence = cached_episodes(signature, [source], load)
    dataset = metadata['dataset_class'].__new__(metadata['dataset_class'])
    vars(dataset).update(metadata['attributes'])
    dataset.episodes = episodes
    dataset.get_episode_iterator = MethodType(private_episode_iterator, dataset)
    return dataset, evidence
