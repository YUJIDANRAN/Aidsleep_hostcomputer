# HostComputer6 PyInstaller 打包说明

本工程可复用 `pyqt_pyinstaller_packaging_guide.md` 里的做法：使用 PyInstaller 的 spec 文件，以 onedir 目录模式打包 PyQt 桌面程序。

## 本工程适配点

- 入口文件：`main.py`
- spec 文件：`packaging/HostComputer6.spec`
- 输出目录：`dist/HostComputer6/HostComputer6.exe`
- 分发内容：复制整个 `dist/HostComputer6/` 目录，不要只复制单个 exe
- 源码搜索路径：工程根目录、`View/`、`Algorithm/`、`Controller/`
- 数据文件：当前只打包 `View/MainWindow.ui`
- 运行时输出：`Result/` 是用户运行后生成的数据，不打包进安装包

## 打包命令

在工程根目录执行：

```powershell
pyinstaller --noconfirm --clean packaging\HostComputer6.spec
```

如果使用本工程虚拟环境：

```powershell
.\.venv\Scripts\pyinstaller.exe --noconfirm --clean packaging\HostComputer6.spec
```

如果当前环境没有 PyInstaller：

```powershell
pip install pyinstaller pyinstaller-hooks-contrib
```

## 验证命令

打包完成后运行：

```powershell
.\dist\HostComputer6\HostComputer6.exe
```

如果 exe 双击无响应，可临时把 `packaging/HostComputer6.spec` 里的 `console=False` 改为 `console=True`，重新打包后从 PowerShell 启动 exe 查看异常。

## 常见补充

如果运行时报 `ModuleNotFoundError`，把缺失模块加入 `hiddenimports`。

如果运行时报 Qt platform plugin 问题，优先升级：

```powershell
pip install --upgrade pyinstaller pyinstaller-hooks-contrib
```

如果界面或资源没有更新，使用当前文档里的 `--clean` 命令重新打包，必要时手动删除 `build/` 和 `dist/` 后再打。
