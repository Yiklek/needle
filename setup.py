#!/usr/bin/env python3
import os
import pathlib
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, path, cmake_args=None, build_args=None, install_args=None, install_to="{lib}"):
        super().__init__(name, sources=[])
        self.path = pathlib.Path(path).absolute()
        self.cmake_args = cmake_args or []
        self.build_args = build_args or []
        self.install_args = install_args or []
        self.install_to = install_to

    def build(self, ctx: "BuildExt"):
        build_temp = pathlib.Path(ctx.build_temp) / self.name
        os.makedirs(build_temp, exist_ok=True)
        ext_path = pathlib.Path(ctx.get_ext_fullpath(self.name))

        config = "Debug" if ctx.debug else "RelWithDebInfo"
        cmake_args = ["-DCMAKE_BUILD_TYPE=" + config, "-GNinja"] + self.cmake_args

        build_args = [
            "--config",
            config,
        ] + self.build_args

        install_args = ["--prefix", self.install_to.format(lib=ctx.build_lib, temp=ctx.build_temp)] + self.install_args

        if not ctx.dry_run:
            ctx.spawn(["cmake", "-S", self.path, "-B", build_temp] + cmake_args)
            ctx.spawn(["cmake", "--build", build_temp] + build_args)
            ctx.spawn(["cmake", "--install", build_temp] + install_args)
            ctx.execute(os.remove, (ext_path,))


class XMakeExtension(Extension):
    def __init__(self, name, path, config_args=None, build_args=None, install_args=None, install_to="{lib}"):
        super().__init__(name, sources=[])
        self.path = pathlib.Path(path).absolute()
        self.config_args = config_args or []
        self.build_args = build_args or []
        self.install_args = install_args or []
        self.install_to = install_to

    def build(self, ctx: "BuildExt"):
        build_temp = pathlib.Path(ctx.build_temp) / self.name
        os.makedirs(build_temp, exist_ok=True)
        ext_path = pathlib.Path(ctx.get_ext_fullpath(self.name))

        config = "debug" if ctx.debug else "release"
        config_args = [
            f"--mode={config}",
            f"--buildir={build_temp}",
        ] + self.config_args

        build_args = self.build_args

        install_args = [
            "--installdir=" + self.install_to.format(lib=ctx.build_lib, temp=ctx.build_temp)
        ] + self.install_args

        if not ctx.dry_run:
            ctx.spawn(["xmake", "config", "-P", self.path] + config_args)
            ctx.spawn(["xmake", "build", "-P", self.path] + build_args)
            ctx.spawn(["xmake", "install", "-P", self.path] + install_args)
            ctx.execute(os.remove, (ext_path,))


class BuildExt(build_ext):
    """
    自定义了 build_ext 类，对 CMakeExtension 的实例，调用 CMake 和 Make 命令来编译它们
    """

    def run(self):
        super().run()
        for ext in self.extensions:
            if isinstance(ext, (CMakeExtension, XMakeExtension)):
                ext.build(self)


core_lib = CMakeExtension("core_lib", ".", build_args=["--target", "backend"], install_args=["--component", "backend"])
# core_lib = XMakeExtension("core_lib", ".", build_args=["--group=backend"],install_args=["--group=backend"])

setup(
    name="needle",
    version="0.1",
    author="Yiklek",
    author_email="yiklek.me@gmail.com",
    description="dlsys framework",
    # 项目主页
    url="https://github.com/Yiklek/needle",
    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages("python"),
    # packages=find_packages(),
    include_package_data=True,
    package_dir={"": "python"},
    classifiers=[
        # 发展时期,常见的如下
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # 开发的目标用户
        "Intended Audience :: Developers",
        # 属于什么类型
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # 许可证信息
        "License :: OSI Approved :: MIT License",
        # 目标 Python 版本
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    # setup_requires=["pbr"],
    # pbr=True,
    python_requires=">=3.8",
    install_requires=["numpy"],
    tests_require=[
        "pytest>=3.3.1",
        "pytest-cov>=2.5.1",
    ],
    ext_modules=[core_lib],  # mymath 现在是 CMakeExtension 类的实例了
    cmdclass={"build_ext": BuildExt},  # 使用自定义的 build_ext 类
)
