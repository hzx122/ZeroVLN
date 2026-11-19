"""Reclaim dead CPU allocations at quiescent payload ownership boundaries."""
import ctypes
import gc
import os


def release_cpu_memory():
    def rss():
        with open('/proc/self/statm') as handle:
            return int(handle.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    before = rss()
    gc.collect()
    trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
    if trim is not None:
        trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
        trim(0)
    return dict(pid=os.getpid(), rss_before_bytes=before, rss_after_bytes=rss(),
                malloc_trim_available=trim is not None)
