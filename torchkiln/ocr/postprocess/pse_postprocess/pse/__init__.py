# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
import subprocess

python_path = sys.executable

# 就地编译 C 扩展 pse：用列表参数 + cwd= 指定工作目录（shell=False）。
# 相比原来的 shell=True + os.chdir 相对路径：避免 shell 注入风险、避免
# Windows cmd 下 POSIX 相对路径脆弱、且不再改动进程 cwd（无全局副作用）。
_pse_dir = os.path.dirname(os.path.realpath(__file__))
if subprocess.call(
        [python_path, "setup.py", "build_ext", "--inplace"],
        cwd=_pse_dir) != 0:
    raise RuntimeError(
        'Cannot compile pse in {}: if your system is Windows, you need to '
        'install all the default components of `desktop development using C++` '
        'in Visual Studio 2019+ '
        '（Windows 需 VS2019+ 的 C++ 桌面开发工具链，含 C++ 编译器与 v143 工具集）。'
        .format(_pse_dir))

from .pse import pse
