"""Linux write sandbox: all inherited mounts read-only, one writable workspace.

Reads and networking remain available. /proc is masked to prevent reaching a
host process's writable root/fds. Capability/namespace/ptrace escape routes are
blocked before exec. Unsupported mount/user namespace setups fail closed.
"""
import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import subprocess
import sys

from workspace import LOGICAL_WORKSPACE


def run(*argv):
    subprocess.run(argv, check=True, stdin=subprocess.DEVNULL)


def enter_user_namespace(command):
    ack_r, ack_w = os.pipe()
    ready_r, ready_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ack_w)
        os.close(ready_r)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.unshare(0x10000000) != 0:
            os._exit(120)
        os.write(ready_w, b'1')
        os.close(ready_w)
        if os.read(ack_r, 1) != b'1':
            os._exit(121)
        os.close(ack_r)
        os.execvp('unshare', ['unshare', '--mount', '--propagation', 'private',
                             '--pid', '--fork', '--kill-child', *command])
    os.close(ack_r)
    os.close(ready_w)
    try:
        if os.read(ready_r, 1) != b'1':
            raise RuntimeError('user namespace unavailable; refusing unsandboxed agent')
        # Existing DSW/DLC parent has this mapping and SETUID/SETGID. Mapping is
        # private; no host accounts or subuid files are changed.
        for name in ('uid_map', 'gid_map'):
            Path(f'/proc/{pid}/{name}').write_text('0 0 4294967295\n')
        os.write(ack_w, b'1')
    finally:
        os.close(ack_w)
        os.close(ready_r)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


def lock_privileges():
    libc = ctypes.CDLL(None, use_errno=True)
    def prctl(option, value):
        if libc.prctl(option, value, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'prctl sandbox setup')
    prctl(38, 1)  # PR_SET_NO_NEW_PRIVS
    prctl(28, 3)  # SECBIT_NOROOT | SECBIT_NOROOT_LOCKED
    for cap in range(41):
        prctl(24, cap)  # PR_CAPBSET_DROP
    libcap = ctypes.CDLL('libcap.so.2')
    libcap.cap_init.restype = ctypes.c_void_p
    libcap.cap_set_proc.argtypes = [ctypes.c_void_p]
    libcap.cap_free.argtypes = [ctypes.c_void_p]
    caps = libcap.cap_init()
    if libcap.cap_set_proc(caps) != 0:
        raise RuntimeError('could not drop process capabilities')
    libcap.cap_free(caps)
    sec = ctypes.CDLL('libseccomp.so.2')
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = sec.seccomp_init(0x7fff0000)  # ALLOW
    if not ctx:
        raise RuntimeError('seccomp initialization failed')
    try:
        for name in ('mount', 'umount2', 'pivot_root', 'chroot', 'unshare', 'setns',
                     'ptrace', 'process_vm_writev', 'open_by_handle_at', 'fsopen',
                     'fsconfig', 'fsmount', 'fspick', 'move_mount', 'open_tree',
                     'mount_setattr', 'bpf', 'io_uring_setup', 'kexec_load', 'reboot'):
            number = sec.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and sec.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, number, 0) != 0:
                raise RuntimeError('seccomp rule failed: ' + name)
        if sec.seccomp_load(ctx) != 0:
            raise RuntimeError('seccomp load failed')
    finally:
        sec.seccomp_release(ctx)


def child(workspace, command):
    workspace = Path(workspace).resolve(strict=True)
    target = Path(LOGICAL_WORKSPACE)
    if not target.is_dir():
        raise ValueError('canonical logical workspace must exist before sandbox launch')
    mounts = {}
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        name = line.split()[4]
        for escape, value in (('\\040', ' '), ('\\011', '\t'), ('\\134', '\\')):
            name = name.replace(escape, value)
        mounts[name] = line.split()[5]
    # Mask foreign process fd/root magic links, then prevent remount/namespace
    # escapes. Regular host files and the network remain readable/reachable.
    run('mount', '-t', 'tmpfs', '-o', 'size=1m,mode=555', 'tmpfs', '/proc')
    mounts['/proc'] = 'rw'
    for mount in sorted(mounts, key=lambda p: len(p), reverse=True):
        if mount.startswith('/proc/'):
            continue
        if not os.statvfs(mount).f_flag & os.ST_RDONLY:
            flags = [flag for flag in mounts[mount].split(',') if flag not in ('rw', 'ro')]
            run('mount', '-o', ','.join(['remount', 'bind', 'ro', *flags]), mount)
    run('mount', '--bind', str(workspace), str(target))
    run('mount', '-o', 'remount,bind,rw', str(target))
    os.chdir(target)
    lock_privileges()
    # agent tmp/cache/home are inside the only writable mount.
    runtime = target / '.runtime'
    runtime.mkdir(exist_ok=True)
    for name in ('home', 'tmp', 'cache'):
        (runtime / name).mkdir(exist_ok=True)
    (runtime / 'bin').mkdir(exist_ok=True)
    for name in ('fd', 'rg'):
        target = runtime / 'bin' / name
        if not target.exists():
            target.symlink_to('/opt/external/user-d5121be6/tools/pi-search/linux-x64/' + name)
    env = dict(os.environ, HOME=str(runtime / 'home'), TMPDIR=str(runtime / 'tmp'),
               XDG_CACHE_HOME=str(runtime / 'cache'), PYTHONDONTWRITEBYTECODE='1',
               PI_CODING_AGENT_DIR=str(runtime))
    for key in ('HF_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH',
                'PYTORCH_KERNEL_CACHE_PATH', 'TORCH_EXTENSIONS_DIR'):
        env[key] = str(runtime / 'cache' / key.lower())
    os.execvpe(command[0], command, env)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('agent command required')
    if args.child:
        child(args.workspace, command)
    else:
        return enter_user_namespace([sys.executable, '-B', str(Path(__file__).resolve()),
                                     '--child', '--workspace', args.workspace, '--', *command])


if __name__ == '__main__':
    raise SystemExit(main())
