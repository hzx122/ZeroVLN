# -*- coding: utf-8 -*-
"""vln_gym — vln_demo 的 gym 风格 HTTP 闭环 RL 环境包。

子模块：
    env.py      VLNEnv（duck-typed gym Env，纯 sim 不含 vLLM）
    fm_action.py (dx,dy,dyaw) 反归一化 + turn→fwd→turn 分解
    server.py   stdlib HTTP gym server（单 session）
    client.py   VLNRemoteClient + smoke

启动：见 ../../vln_gym_server.sh
文档：见 ../../docs/vln_demo的gym风格HTTP接口与闭环RL环境说明.md
"""
