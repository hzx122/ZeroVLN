"""Bypass external proxies only for the explicitly bound private services."""
from urllib.parse import urlparse


def service_env(env, *urls):
    result = dict(env)
    hosts = ['localhost', '127.0.0.1', '::1']
    hosts += [urlparse(url).hostname for url in urls if url and urlparse(url).hostname]
    existing = (env.get('NO_PROXY', '')+','+env.get('no_proxy', '')).split(',')
    result['NO_PROXY'] = result['no_proxy'] = ','.join(dict.fromkeys(x for x in hosts+existing if x))
    return result
