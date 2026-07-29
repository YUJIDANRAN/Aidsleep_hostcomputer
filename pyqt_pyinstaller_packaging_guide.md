# PyQt 软件 PyInstaller 打包复用指南

本文档整理自 `AudioCreator` 工程当前的打包方式，可迁移到新的 PyQt/PyQtGraph/Matplotlib 桌面软件工程。

## 当前工程打包方式

当前工程使用 **PyInstaller + spec 文件** 打包，输出形式是 **onedir 目录模式**，也就是生成一个可执行文件和一组依赖文件目录。

当前打包入口：

```text
main.py
```

当前 spec 文件：

```text
packaging/AudioCreator.spec
```

当前打包命令：

```powershell
pyinstaller --noconfirm --clean packaging\AudioCreator.spec
```

如果使用项目虚拟环境：

```powershell
.\.venv\Scripts\pyinstaller.exe --noconfirm --clean packaging\AudioCreator.spec
```

输出目录：

```text
dist\AudioCreator\AudioCreator.exe
dist\AudioCreator\_internal\
```

分发时应复制整个目录：

```text
dist\AudioCreator\
```

不要只复制单个 `.exe`，因为 `_internal` 中包含 Python、PyQt、NumPy、SciPy、Matplotlib 等运行依赖。

## 参数含义

`--noconfirm`：

```text
覆盖已有 build/dist 输出，不再交互确认。
```

`--clean`：

```text
打包前清理 PyInstaller 缓存和中间构建文件，减少旧缓存导致的资源或代码未更新问题。
```

## 当前 spec 的关键配置

当前工程 spec 的核心思路：

1. 通过 `PROJECT_ROOT` 定位工程根目录。
2. 把 `main.py` 作为程序入口。
3. 把 `src` 加入 `pathex`，使源码包可被 PyInstaller 分析。
4. 把 `resources` 和 `outputs` 作为数据目录打入包内。
5. 手动补充 PyInstaller 不容易自动发现的 hidden imports。
6. 使用 `console=False` 生成无控制台窗口的 GUI 程序。
7. 使用 `COLLECT` 生成 onedir 目录。

当前工程相关依赖：

```text
PyQt6
numpy
scipy
matplotlib
pyqtgraph
soundfile
pydub
pyserial
```

当前 hidden imports：

```python
hiddenimports = [
    "pyqtgraph",
    "matplotlib.backends.backend_qtagg",
    "scipy.io.wavfile",
    "soundfile",
]
```

当前 excludes：

```python
excludes=["pyqtgraph.opengl", "OpenGL"]
```

这表示项目不使用 PyQtGraph OpenGL 功能，因此排除 OpenGL 相关模块以减少体积和打包问题。

## 可复用 spec 模板

新工程可以复制下面模板，按注释替换项目名、入口文件、资源目录和 hidden imports。

```python
# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


# 如果 spec 文件放在 packaging/ 下：
# SPECPATH = <project>/packaging
# Path(SPECPATH).parent = <project>
PROJECT_ROOT = Path(SPECPATH).parent


APP_NAME = "YourAppName"
ENTRY_SCRIPT = PROJECT_ROOT / "main.py"
SRC_DIR = PROJECT_ROOT / "src"


datas = []
for name in ("resources",):
    path = PROJECT_ROOT / name
    if path.exists():
        datas.append((str(path), name))


hiddenimports = [
    # PyQtGraph / Matplotlib 常见补充项
    "pyqtgraph",
    "matplotlib.backends.backend_qtagg",
]


a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 不用 OpenGL 时可排除
        "pyqtgraph.opengl",
        "OpenGL",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
```

## 推荐工程结构

建议新工程使用类似结构：

```text
NewPyQtProject/
  main.py
  requirements.txt
  packaging/
    NewApp.spec
  src/
    your_package/
      __init__.py
      ui/
      core/
  resources/
    icons/
    images/
  docs/
```

入口 `main.py` 推荐写法：

```python
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from your_package.ui.main_window import run_app


if __name__ == "__main__":
    raise SystemExit(run_app())
```

这种写法的好处是：

```text
开发环境直接 python main.py 可以运行；
PyInstaller 打包后也能找到 src 包。
```

## 迁移步骤

1. 在新工程中安装 PyInstaller：

```powershell
pip install pyinstaller
```

2. 确认程序能从源码运行：

```powershell
python main.py
```

3. 新建目录：

```text
packaging/
```

4. 复制 spec 模板并重命名：

```text
packaging/NewApp.spec
```

5. 修改 spec 中的：

```text
APP_NAME
ENTRY_SCRIPT
SRC_DIR
datas
hiddenimports
excludes
```

6. 执行打包：

```powershell
pyinstaller --noconfirm --clean packaging\NewApp.spec
```

7. 运行输出程序：

```text
dist\NewApp\NewApp.exe
```

8. 若运行报缺模块，再回到 spec 中补充 `hiddenimports`。

## datas 配置建议

`datas` 用于打包非 Python 文件，例如：

```text
图片
图标
配置文件
样式表
模型文件
默认音频/示例数据
```

示例：

```python
datas = []
for name in ("resources", "config"):
    path = PROJECT_ROOT / name
    if path.exists():
        datas.append((str(path), name))
```

注意：

```text
不建议把用户运行时输出目录打包进去，除非确实需要内置示例输出。
```

当前 `AudioCreator` 把 `outputs` 也打包进去了，是因为项目里已有默认生成音频和 mapping 示例。新工程迁移时应按实际情况决定。

## 常见 hidden imports

PyInstaller 有时无法自动分析动态导入模块。常见补充项：

```python
hiddenimports = [
    "pyqtgraph",
    "matplotlib.backends.backend_qtagg",
    "scipy.io.wavfile",
    "soundfile",
]
```

如果使用 PyQt6 多媒体：

```text
一般 PyInstaller 会通过 hook 自动收集 Qt 插件；
若运行时报 Qt platform plugin 或 multimedia plugin 问题，优先升级 pyinstaller 和 pyinstaller-hooks-contrib。
```

```powershell
pip install --upgrade pyinstaller pyinstaller-hooks-contrib
```

## console 选项

GUI 软件通常使用：

```python
console=False
```

这样启动时不会弹出黑色控制台窗口。

调试打包问题时可以临时改成：

```python
console=True
```

这样运行 exe 时可以看到 Python 异常输出。

## onedir 与 onefile

当前工程使用 onedir：

```text
dist/AppName/AppName.exe
dist/AppName/_internal/
```

优点：

```text
启动快
依赖清晰
适合 PyQt、Matplotlib、SciPy 这类大型 GUI 软件
问题更容易排查
```

onefile 虽然只有一个 exe，但启动时需要解压依赖，PyQt 大型项目更容易慢或出问题。推荐优先使用 onedir。

## 常见问题排查

### 1. 源码能运行，exe 打不开

临时改：

```python
console=True
```

重新打包，然后从 PowerShell 运行 exe 看报错。

### 2. ModuleNotFoundError

把缺失模块加到：

```python
hiddenimports = [...]
```

### 3. 资源文件找不到

确认资源目录已经加入：

```python
datas = [...]
```

同时程序中不要写死开发环境绝对路径。推荐封装资源路径函数，兼容源码运行和 frozen 运行。

### 4. 打包后还是旧界面

使用：

```powershell
pyinstaller --noconfirm --clean packaging\AppName.spec
```

必要时删除旧的：

```text
build/
dist/
```

### 5. 打包体积过大

可考虑：

```text
排除未使用模块
减少 datas 内容
不用 onefile
检查是否误打包大量输出数据
```

## 打包前检查清单

打包前：

```text
[ ] python main.py 可以正常运行
[ ] requirements.txt 完整
[ ] spec 中 APP_NAME 正确
[ ] spec 中入口 main.py 正确
[ ] spec 中 pathex 包含 src
[ ] 必需资源目录已加入 datas
[ ] 不需要的输出/临时文件没有误加入 datas
[ ] GUI 程序使用 console=False
```

打包后：

```text
[ ] dist\AppName\AppName.exe 可以启动
[ ] 图标、图片、配置文件正常加载
[ ] PyQt 页面能正常打开
[ ] Matplotlib/PyQtGraph 图能正常显示
[ ] 串口、音频、文件导入等外设功能按需验证
```

## 可转成 Codex Skill 的结构

如果要把本流程沉淀为个人 skill，可以按下面结构创建：

```text
pyqt-pyinstaller-packager/
  SKILL.md
  references/
    pyinstaller_spec_template.py
    troubleshooting.md
```

`SKILL.md` 可包含：

```text
适用场景：
- 用户需要打包 PyQt/PySide 桌面软件
- 用户需要迁移 PyInstaller spec
- 用户遇到 PyQt 打包后资源丢失、hidden import、Qt plugin 问题

工作流程：
1. 读取 main.py、requirements.txt、现有 spec
2. 确认入口、src 包路径、资源目录
3. 生成或修改 packaging/AppName.spec
4. 给出打包命令
5. 检查 dist 输出
6. 根据运行错误补 hiddenimports/datas/excludes
```

是否创建 skill 取决于后续是否会频繁处理类似 PyQt 打包任务。如果只是迁移一次，保留本文档即可；如果会在多个工程复用，建议创建 skill。
