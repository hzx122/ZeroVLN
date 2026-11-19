# 前端测试

没有构建步骤，也没有 npm 依赖 —— 只要有 `node`：

```bash
./run.sh          # 用 fixtures/ 里存好的 /config.json 和 /status
./run.sh 5000     # 先从跑着的控制台刷新 fixtures，再跑
```

`shim.mjs` 是一个刚好够把页面启动起来的 DOM。它**故意**在两件事上跟浏览器一致，
因为不一致的 shim 会凭空造出或者藏掉 bug：

- `getElementById` 从 `documentElement` 往下走，**没挂到文档上的元素查不到**。
  面板是先建好、再由 dock 挂上去的，所以一个用 `document.getElementById` 给自己
  接线的面板什么也找不到 —— 这正是第一版 dock 上线后整页卡在「连接中…」的原因
  （`controls.js` 在 build() 里用 `$()` 找自己的下拉框，拿到 null）。现在面板内部
  的查找走 `util.js` 的 `within(root, id)`。
- `<select>.value` 是选中 option 的值，赋一个不存在的值是**空操作**；
  `checked`/`selected`/`disabled` 这类布尔属性，属性存在就等于 true。
  之前把 `value` 当普通字段的写法，让一个键名写错的测试「通过」了。

`querySelector` 那边的选择器只认 `tag` / `.cls` / `#id` 的组合；属性和伪类选择器
**一律不匹配**，而不是猜一个结果 —— 一个判断不了的选择器返回 true 会让断言凭空成立。
（所以 `dock.js` 里那句 `querySelector('[data-nid=…]')` 在 shim 下永远是 null，
最大化的路径只有浏览器里才走得到。）

测试包含页面 JS/CSS 回归和一条服务端图像链路回归：

| 文件 | 查什么 |
|---|---|
| `boot.mjs` | 整页能启动：八个面板都挂上、状态药丸不再是「连接中…」、没有异常 |
| `interact.mjs` | 每个控件在 render 之后都找得到；切资产集/选场景/换驱动/加载/换主题都按预期动 |
| `params.mjs` | `params()` 能从 DOM 把提示词参数全取出来（`instruction` 为空是设计如此） |
| `tabs.mjs` | 「提示词」面板在隐藏标签页时不拉 `/prompt`，激活后才拉 |
| `dock.mjs` | 457 次随机拖放后的不变量：八个面板不重不丢、每个 split 的比例和为 100 且不低于 6%、面板都还挂在文档里；存档能还原；坏 JSON 和过期布局能修 |
| `lock.mjs` | 🔒 锁上之后 `cmd()` / `postSysPrompt()` / 地图双击一条都发不出去，提示是那句原话，状态跟着 `/status.locked` 走（另一个标签页锁/解锁本页也跟） |
| `map.mjs` | 双击俯视地图落在你瞄的那个点：`pictureRect()` 把 `object-fit: contain` 的黑边扣掉，黑边上的双击不发指令 |
| `css.mjs` | 样式表真的按 dock 以为的样子排版：dock 自己的 class 一律 `dk-` 前缀、没有别的样式表给 dock 元素设布局属性、每个 split 算出来的 `flex-direction` 和树里的 `dir` 一致且没有 `flex-wrap:wrap`、两路推流铺满自己的框 |
| `frame_pipeline.py` | 第一视角 Web/observe/模型输入同尺寸且无 step 叠字；bbox 只能使用它框选时的原子帧 |

跑不动 `dock.mjs` 里那些 6% 的断言时先看 `MIN_PCT`：拖放和拽分隔线共用这个下限，
低于它面板的标签条就窄到抓不住，等于把面板弄丢了。

## 为什么要有 css.mjs

前五个测试跑的是 JS，一行 CSS 都没读过 —— 于是 dock 上线时每个横向 split 的 class 是
`"dk-split row"`，同时命中了 base.css 里毫不相干的表单行规则（`flex-wrap: wrap`、
`align-items: center`、`gap: 7px`、`margin-bottom: 8px`）。默认的两列布局因此换行成一
列竖着堆，每个面板都塌成只剩标签条。JS 全对，页面没法用，五个测试全绿。

`css.mjs` 把四份样式表解析成规则表（选择器只支持 `tag` / `#id` / `.cls` 和后代、子代两种
组合符，带伪类/属性/`*` 的一律跳过并**列出来**，不静默），渲染真的 dock，然后对每个 dock
自己的元素算一遍层叠。它守两条：

1. dock 放在自家家具上的 class 全部 `dk-` 前缀 —— `row` / `on` / `dot` 这种短名字属于面板，
   撞了在 JS 里完全看不出来。
2. 除 dock.css 以外的样式表不许给 dock 元素设任何布局属性 —— 这是把上面那次事故写成层叠
   的性质，而不是写成某一个 class 名字。

边界是 `.dk-pane` 和 `.dk-tools` 里的按钮：那是面板自己的元素（log.js 的根就是
`class="dk-pane log-box"`，「清空」按钮也是它给的），panels.css / base.css 管它们是应该的。

它还顺手盯着两路推流：`#rgb` 和 `#map` 必须 `width/height: 100%` + `object-fit: contain`。
`max-width/max-height` 只会缩不会放，图源（RGB 640×480、地图定高 420）比面板小，所以那
个写法会让画面按原始尺寸摆在框中间、只占一半面积。

## 为什么要有 map.mjs

上面那条铺满的代价：`img` 的元素框从此是整个框，画面在里面居中留黑边。而双击传送发的是
**画面**的比例（0..1），服务端拿它反推世界坐标。两个框以前是同一个，`(clientX - r.left) /
r.width` 就够；现在差了一整条黑边，`map.mjs` 钉的就是这个差值 —— 用元素框顶替画面框，
它会以「一格黑边上的双击也发了 goto」和「画面顶端算出 0.033 而不是 0」两条同时失败。
