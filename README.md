# 帧序笔触重建（高精度静态 / 动态分区）

把参考视频在离线阶段拆解为带时间维度的参数化绘图指令，再由浏览器按帧执行指令绘制画面。运行时不加载 mp4、不使用 `<video>`、不依赖视频解码器；源视频只作为预处理输入。

当前版本重新划分了静态区间和需要动态表达的内容。旧版把人物、翅膀等运动对象混入了静态基底，动态区又使用了 8×8 色块；这一版把静态基底改为 RGBA，动态像素透明，并把动态重建提升到 2×2 图元命令加稀疏逐像素修正。

## 高精度分区与绘图指令

离线预处理由 `tools/analyze_video.py` 读取 `C:\Users\vipshao\Desktop\参考.mp4`：

1. 对 124 帧统计每个像素的 2%–98% 时序通道范围和完整时序范围。前者大于 12 或后者大于 16 的像素先标记为动态候选。
2. 对候选区域做 9px 闭运算、大轮廓填充、11px 闭运算、1px 安全膨胀和至少 16px 的扫描线连续段清理，再按 2×2 命令网格对齐。人物、翅膀、粒子和运动轮廓作为连通区域全部进入动态层；稳定背景保留为白色锁定区域。
3. `static-base.png` 是 RGBA 图，动态像素 alpha 为 0；`static-lock.png` 白色像素由固定的 WebGL 基底纹理提供，逐帧不变。`static-partition.json` 记录逐行静态 / 动态区间、连通区域边界和分区规则。
4. 每 4 帧提取参考帧与基底的残差轮廓，转换为椭圆、线段、贝塞尔和笔触参数，记录颜色、中心、半径、旋转、宽度、透明度、位移和形变。匈牙利匹配保持笔触身份，Farneback 光流测量逐时刻运动。`drawing-instructions.json` 保存 38 条轨迹和 344 个关键帧状态。
5. `frame-commands.bin` 使用 SBT4：每 8 帧有一组完整的 2×2 RGB555 图元命令，图元编号由局部边缘方向决定；中间帧只写入变化明显的 2×2 命令区内逐像素重绘指令，位置用 LEB128 差值编码。浏览器从最近关键帧顺序重放到目标帧，因此任意拖动时间轴都能得到同一时刻的绘图状态。静态区从不参与时序更新。
6. 离线逐帧比较重建结果与源画面，结果写入 `quality-report.json`。浏览器页面中的 MAE、动态区 MAE、PSNR、锁定比例和三帧自检均读取这份报告。

浏览器运行时 `src/timeline.js` 在关键帧之间插值，`src/frame-commands.js` 解码并重放 SBT4 指令，`src/renderer.js` 用 Canvas 2D 绘制参数笔触，再用 WebGL2 按 RGBA 基底和锁定遮罩合成。参考帧指令同时修正笔触的中心、半径、角度和颜色。

## 运行

```bash
npm run dev
# http://127.0.0.1:4173
```

```bash
npm test               # 指令、模块、124 帧静态锁定和零运行时视频检查
npm run analyze        # 从桌面源视频重新生成离线指令和报告
npm run build          # 输出可部署的 dist/
npm run bundle:runtime # 仅打包浏览器实际加载的 14 个文件，并生成 ZIP 与校验清单
npm run preview        # 从 dist/ 为根目录预览发布版
```

预处理依赖 Python 3、OpenCV、NumPy、Pillow 和 SciPy。浏览器运行时使用 ES modules、Canvas 2D、WebGL2 和 `DecompressionStream`，不需要第三方库、Python 或 Node。

## 本体依赖资源

浏览器本体依赖资源共 **14 个文件**，原始大小 **19,903,482 字节（约 18.982 MiB）**，其中高精度绘图指令 18,348,880 字节；压缩 ZIP 为 **19,626,339 字节（约 18.718 MiB）**。ZIP SHA-256 为 `97c18ec3bc3a63bacff17d46e4f6578aef6cb857e8b59ae0173a9fc560684d28`。

打包结果位于 `release/runtime-bundle/`，ZIP 为 `release/runtime-bundle.zip`，逐文件字节数与 SHA-256 位于 `release/runtime-bundle-manifest.json`。包内不含视频、Python、Node、测试、分析脚本、README、环境配置或示例图；部署时需要能加载静态文件的 HTTP 服务和支持上述浏览器 API 的现代浏览器。

## 时间轴与视图

- 时间轴支持逐帧跳转、播放和速率切换，帧号与 24 fps 时间同步计算。
- 六个视图：复原合成、指令参考、参数笔触层、锁定基底、参考差分、分屏比对。
- “运行 3 帧同步验证”检查 F000、F060、F120 的帧时间、活动笔触、绘图调用和像素指标。
- “指令帧驱动形状”“锁定 WebGL 基底”“笔触贡献”可分别放开对应约束，用于比较形变和笔触造型。

## 验证结果

- 连通分区锁定像素 **50.1%**，动态表达 **49.9%**；对 124 帧逐帧检查，锁定区域帧间差为 `0.000`，最大平均残差 `1.401/255`。锁定基底视图中人物与翅膀为黑色空洞，由动态指令层重建。
- 离线质量报告：平均全画面 MAE **1.820/255**，最差 MAE **2.819/255**，最低 PSNR **32.50 dB**，平均动态区 MAE **2.655/255**。
- `frame-commands.bin` 为 **18,348,880 字节**，平均每帧 41,351.9 个命令区指令；较 8×8 版本的 MAE 5.045 和 PSNR 22.95 dB 有显著提升。
- 浏览器三帧自检为 **PASS**：F000、F060、F120 分别有 12、14、11 条活动笔触，132,011、72,081、132,007 个绘图调用，最大同步偏差 `0.0 ms`，静态帧间差 `0.000`。
- 资源校验确认 `public/assets/`、`dist/`、运行时包和 ZIP 中均无 mp4；页面无 `<video>` 元素，也没有视频媒体运行时代码。

## 仓库内容

```text
index.html                     应用入口
src/                           时间轴、SBT4 指令重放、WebGL 合成、指标与界面
tools/analyze_video.py         离线视觉预处理（读取本地源视频）
tools/compact_commands.py      高精度静态分区与 SBT4 绘图指令编码
tools/serve.mjs                本地静态服务器
tools/build.mjs                dist 构建
tools/validate-*.mjs/py        数据、基底和运行时资源验证
public/assets/                 基底、遮罩、绘图指令和离线报告
analysis/                      本地诊断输出目录（不发布）
```

在线页面：<https://vipshao-arch.github.io/reference-driven-stroke-reconstruction/>

仓库：<https://github.com/vipshao-arch/reference-driven-stroke-reconstruction>
