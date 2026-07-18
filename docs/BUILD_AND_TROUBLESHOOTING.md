# SConvTransform 构建与排错记录

> 在 aarch64 (Apple Silicon / Linux ARM64) 上从零构建并运行 SConvTransform
> 遇到的所有问题、原因、修复, 以及在 **非 macOS 的 aarch64 Linux** 上复现的指南。

---

## 1. 环境概览

| 项目 | 值 |
|------|-----|
| 宿主 OS | macOS Apple Silicon (测试) / Linux aarch64 (复现目标) |
| 项目 | SConvTransform (MLIR Transform Dialect 扩展) |
| 最终使用 LLVM | **19.1.7** (预编译 release) |
| 也测试过 LLVM | 20.1.5 (自编译, 编译通过但运行时 API 不兼容) |
| 编译器 | Apple Clang 21 / GCC 13+ (Linux) |
| 构建系统 | CMake >= 3.22 + Ninja |
| 构建类型 | Release + Assertions ON |

## 2. LLVM 版本选择

开发周期 2024-09 ~ 2025-11。代码使用 `linalg::splitOp` (LLVM 19)、
`registerAllExtensions` (LLVM 19)、`scf::SCFTilingResult` (LLVM 18+) 等 API,
最低要求 LLVM 19。

| 版本 | 结论 | 说明 |
|------|------|------|
| **LLVM 19.1.x** | **推荐** | API 匹配, 需 3 处小适配 (见下) |
| LLVM 20.1.x | 编译通过, 运行时失败 | TransformResults 变长索引变了 (问题 6) |
| LLVM <= 18 | 不可用 | 缺关键 API |

> Linux aarch64: 从 GitHub `llvm-project/releases` 下载
> `LLVM-19.1.7-aarch64-linux-gnu.tar.xz`, 或从 `release/19.1.7` tag 自编译。

---

## 问题 1: lld 链接器不存在

**现象**: 链接时报错 `cannot find ld.lld`。

**原因**: `CMakeLists.txt` 第 15-17 行硬编码 `-fuse-ld=lld`。lld 是 LLVM
链接器, macOS 上不存在 (用 Apple `ld64`); Linux 上需单独安装。

**修复**: `CMakeLists.txt` 用 `if(NOT APPLE)` 守卫:
```cmake
if(NOT APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=lld")
endif()
```

> Linux aarch64: 装了 `lld` 不用改; 没装则 `apt install lld` / `dnf install lld`。

---

## 问题 2: C++ 标准未设置

**现象**: 大量 `no template named 'is_class_v' in namespace 'std'` 错误。

**原因**: 项目未设 `CMAKE_CXX_STANDARD`。LLVM 头文件用 C++17 变量模板
(`std::is_class_v` 等), 但编译器默认 C++14。

**修复**: `configure.sh` 加 `-DCMAKE_CXX_STANDARD=17`。

> Linux aarch64: 同样需要, GCC 默认也可能不是 C++17。

---

## 问题 3: Build 类型不匹配

**现象**: 链接 ABI 不匹配或 assertion 失败。

**原因**: 原 `configure.sh` 用 `Debug`, 但 LLVM release 是 `Release`+assertions。
out-of-tree 项目须与 LLVM build 的 `NDEBUG`/assertion 模式一致。

**修复**: `configure.sh` 改为:
```bash
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_CXX_STANDARD=17
-DLLVM_ENABLE_ASSERTIONS=ON
```

> 如果自编译 LLVM 用了 `RelWithDebInfo`, 这里也设 `RelWithDebInfo`。

---

## 问题 4: setAttrs 初始化列表语法不兼容

**现象**: `no matching member function for call to 'setAttrs'`。

**原因**: `lib/SConv.cpp` 中 5 处用 `op->setAttrs({{"k", attr}, ...})` 初始化列表。
LLVM 19 的 `setAttrs` 只接受 `DictionaryAttr` 或 `ArrayRef<NamedAttribute>`,
不接受裸 `{{...}}` 列表。LLVM 20 能通过隐式转换编译。

**修复**: 拆成逐个 `setAttr` 调用:
```cpp
// 改前:
op->setAttrs({{"packing", rewriter.getStringAttr("filter")},
              {"multipacking", rewriter.getBoolAttr(false)}});
// 改后:
op->setAttrs(rewriter.getNamedAttr("packing", rewriter.getStringAttr("filter")));
op->setAttr("multipacking", rewriter.getBoolAttr(false));
```

涉及 5 处: `applyFilterPacking`(:836), `applyInputPacking`(:980),
`inputMultipackingOpt`(:1199), `filterMultipackingOpt`(:1449), `applyTileTo`(:1732)。

---

## 问题 5: getStridesAndOffset 成员 -> 自由函数

**现象**: `no member named 'getStridesAndOffset' in 'mlir::MemRefType'`。

**原因**: `lib/LowerToBLAS.cpp:176` 用了 `outTileTy.getStridesAndOffset()`
(成员函数调用)。LLVM 19 中 `getStridesAndOffset` 是**自由函数**
`mlir::getStridesAndOffset(MemRefType)`, 返回
`std::pair<SmallVector<int64_t>, int64_t>`。

**修复**:
```cpp
// 改前:
auto [strides, _] = outTileTy.getStridesAndOffset();
// 改后:
SmallVector<int64_t> strides;
int64_t offset;
std::tie(strides, std::ignore) = mlir::getStridesAndOffset(outTileTy);
```

---

## 问题 6: LLVM 20 TransformResults API 变更

**现象**: sconv-opt 运行时 assertion 失败:
`querying unset results (values or params expected?)`

**原因**: LLVM 20 的 `TransformResults` 索引从"按声明组"变为"按实际 OpResult"。

SConvOp 定义了 `Variadic<TransformHandleTypeInterface>:$loops`, 测试文件用
`%loops:6` (6 个变长结果)。LLVM 20 中
`TransformResults(transform->getNumResults())` 创建 **7 个 slot**
(1 output_convs + 6 variadic), 但 `apply()` 只 set 了 slot 0 和 1,
slot 2-6 全空 -> assertion。

LLVM 19 中 slot 数量 = 声明组数 = **2**, slot 1 覆盖整个 variadic,
框架自动分发到 6 个结果。

**修复**: **换用 LLVM 19.1.7**。若需适配 LLVM 20, 需改 `apply()` 逐个 set。

---

## 问题 7: swapInductionVars SSA 支配违规

**现象**: transform 后的 MLIR 文件被 `mlir-opt` 读取时报错:
`use of undeclared SSA value name`。`scf.for` 的循环界常量
(%54, %55, %56) 定义在循环体**内部**, 但被循环**自身**引用。

**原因**: `swapInductionVars()` (`lib/SConv.cpp:1056-1071`) 交换内外循环界:
```cpp
outerLoop.setLowerBound(innerLowerBound);  // inner 界来自 outer body 内部
```
`scf::tileUsingSCF` 生成的内循环界常量 (0, 1024, 16) 定义在**外循环 body** 中。
交换后, 外循环引用了定义在自身 body 内的值, 违反 SSA 支配关系。

**修复**: 交换前先把内循环界常量提升到外循环**之前** (`lib/SConv.cpp:1062-1074`):
```cpp
auto hoistIfLocal = [&](Value &v) {
  if (auto *defOp = v.getDefiningOp()) {
    if (defOp->getBlock() == innerLoop.getBody() ||
        defOp->getBlock() == outerLoop.getBody())
      defOp->moveBefore(outerLoop);
  }
};
hoistIfLocal(innerLowerBound);
hoistIfLocal(innerUpperBound);
hoistIfLocal(innerStep);
```

这是原始代码的 bug (在 LLVM 19 上也会触发), 不是版本适配问题。

---

## 问题 8: macOS Gatekeeper 隔离

**现象**: 从 llvm.org 下载的预编译 LLVM, 运行 `llvm-config` / `mlir-opt` 时
`exit=137` (SIGKILL)。

**原因**: 下载的 tar 包带 `com.apple.quarantine` 扩展属性, macOS Gatekeeper
拦截执行。

**修复**:
```bash
xattr -rd com.apple.quarantine /path/to/LLVM-19.1.7-macOS-ARM64
```

> **仅 macOS 需要处理**。Linux 上无此问题。但如果 tar 包里有 macOS 的
> `._` (AppleDouble) 文件干扰, 可用 `find . -name '._*' -delete` 清理。

---

## 问题 9: mlir-runner 不存在

**现象**: `run.sh` 调用 `mlir-runner` 但找不到该命令。

**原因**: LLVM 19 release 预编译包只有 `mlir-cpu-runner`, 没有 `mlir-runner`。
`mlir-runner` 是 LLVM 20+ 引入的新名称 (替代 `mlir-cpu-runner`)。

**修复**: 用 `mlir-cpu-runner` 替代:
```bash
mlir-cpu-runner $OUTPUT -e main -entry-point-result=void \
  -shared-libs="$LLVM_LIB_PATH/libmlir_c_runner_utils.dylib,$LLVM_LIB_PATH/libmlir_runner_utils.dylib"
```

> Linux aarch64: 同样, LLVM 19 预编译包只有 `mlir-cpu-runner`。
> 如果自编译 LLVM 19 且在 cmake 时启用了 `MLIR_CPU_RUNNER`, 也会产出此工具。

---

## 问题 10: LLVM 源码树缺失文件

**现象**: 编译时报 `fatal error: 'mlir/Dialect/Polynomial/IR/PolynomialDialect.h'
file not found`。

**原因**: 如果 LLVM 是从 git 浅克隆后, 某些目录/文件可能被意外删除
(`git status` 显示 ` D` 状态)。`mlir/InitAllDialects.h` 引用了
`Polynomial` 方言头文件, 但 `mlir/include/mlir/Dialect/Polynomial/` 目录缺失。

**修复**:
```bash
cd /path/to/llvm-project
git checkout -- mlir/include/mlir/Dialect/Polynomial/
git checkout -- mlir/lib/Dialect/Polynomial/
```
或者批量恢复所有被删文件:
```bash
git status --short -- mlir/ | grep "^ D" | awk '{print $2}' | xargs git checkout --
```

> **仅影响自编译 LLVM 且手动清理过源码的情况**。预编译 release 包不受影响
> (头文件完整)。Linux 上若用预编译包, 不会遇到此问题。

---

## Linux aarch64 复现指南

### 步骤 1: 获取 LLVM 19.1.7

```bash
# 方式 A: 下载预编译包 (推荐, 快)
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-19.1.7/LLVM-19.1.7-aarch64-linux-gnu.tar.xz
tar xf LLVM-19.1.7-aarch64-linux-gnu.tar.xz
export LLVM_BUILD_DIR="$(pwd)/LLVM-19.1.7-aarch64-linux-gnu"

# 方式 B: 从源码编译 (慢, 但可定制)
git clone --depth 1 --branch llvmorg-19.1.7 https://github.com/llvm/llvm-project.git
cd llvm-project
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD="AArch64;X86" \
  -DLLVM_ENABLE_LLD=ON
cmake --build build -j$(nproc)
export LLVM_BUILD_DIR="$(pwd)/build"
```

### 步骤 2: 安装依赖

```bash
# Ubuntu/Debian
sudo apt install cmake ninja-build clang lld

# Fedora/RHEL
sudo dnf install cmake ninja-build clang lld
```

### 步骤 3: 配置 + 编译 SConvTransform

```bash
cd /path/to/SConvTransform
export LLVM_BUILD_DIR="/path/to/llvm-19.1.7"
make configure   # = ./configure.sh
make build       # = cmake --build build
# 产物: build/bin/sconv-opt
```

### 步骤 4: 运行测试

```bash
# Transform 测试 (6 transform x 3 payload = 18 组合)
export PATH="$LLVM_BUILD_DIR/bin:$PATH"
export LLVM_LIB_PATH="$LLVM_BUILD_DIR/lib"

for t in test/sconv.mlir test/sconv_mK.mlir test/sconv_arch.mlir \
         test/sconv_latency.mlir test/s2conv.mlir test/s2conv_mK.mlir; do
  for p in test/payload.mlir test/pay2load.mlir test/paysload.mlir; do
    ./build/bin/sconv-opt -transform="$t" "$p" > /dev/null 2>&1 && \
      echo "OK: $(basename $t) x $(basename $p)" || \
      echo "FAIL: $(basename $t) x $(basename $p)"
  done
done
```

### 步骤 5: 完整端到端 (transform + lower + JIT)

```bash
# 需要一个带 main 函数的 payload (见 test/payload.mlir + 添加 main 包装)

# 1. Baseline: 无优化, 直接 lower + 跑
mlir-opt payload_main.mlir \
  --one-shot-bufferize='bufferize-function-boundaries' \
  --convert-linalg-to-affine-loops --canonicalize --cse \
  --expand-strided-metadata --lower-affine --convert-scf-to-cf \
  --normalize-memrefs --memref-expand --finalize-memref-to-llvm \
  --lower-affine --convert-func-to-llvm --convert-arith-to-llvm \
  --convert-cf-to-llvm --canonicalize --cse --symbol-dce \
  --llvm-legalize-for-export \
  -o baseline.llvm.mlir

mlir-cpu-runner baseline.llvm.mlir -e main -entry-point-result=void \
  -shared-libs="$LLVM_LIB_PATH/libmlir_c_runner_utils.so,$LLVM_LIB_PATH/libmlir_runner_utils.so"

# 2. SConv transform 后 lower + 跑
./build/bin/sconv-opt -transform=test/sconv_mK.mlir payload_main.mlir > transformed.mlir

mlir-opt transformed.mlir \
  --one-shot-bufferize='bufferize-function-boundaries' \
  --convert-linalg-to-affine-loops --canonicalize --cse \
  --expand-strided-metadata --lower-affine --convert-scf-to-cf \
  --normalize-memrefs --memref-expand --finalize-memref-to-llvm \
  --lower-affine --convert-func-to-llvm --convert-arith-to-llvm \
  --convert-cf-to-llvm --canonicalize --cse --symbol-dce \
  --llvm-legalize-for-export \
  -o transformed.llvm.mlir

mlir-cpu-runner transformed.llvm.mlir -e main -entry-point-result=void \
  -shared-libs="$LLVM_LIB_PATH/libmlir_c_runner_utils.so,$LLVM_LIB_PATH/libmlir_runner_utils.so"

# 3. 对比输出: 两者的数值应该完全一致
```

> 注意: `run.sh` 原脚本用 `mlir-runner` (LLVM 20+), LLVM 19 上需改用
> `mlir-cpu-runner`。动态库后缀 Linux 是 `.so`, macOS 是 `.dylib`。

---

## 改动 Diff 汇总

共改动 4 个文件, 35 行增 / 16 行删:

### CMakeLists.txt (lld 守卫)
```diff
-set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
-set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")
-set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=lld")
+# lld is only available on Linux; skip on macOS (uses Apple's ld).
+if(NOT APPLE)
+  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
+  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")
+  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=lld")
+endif()
```

### configure.sh (build type + C++17)
```diff
-   -DCMAKE_BUILD_TYPE=Debug     \
-   -DLLVM_ENABLE_LLD=ON         \
+   -DCMAKE_BUILD_TYPE=Release  \
+   -DCMAKE_CXX_STANDARD=17     \
+   -DLLVM_ENABLE_ASSERTIONS=ON \
```

### lib/LowerToBLAS.cpp (getStridesAndOffset)
```diff
-  auto [strides, _] = outTileTy.getStridesAndOffset();
+  SmallVector<int64_t> strides;
+  int64_t offset;
+  std::tie(strides, std::ignore) = mlir::getStridesAndOffset(outTileTy);
```

### lib/SConv.cpp (setAttrs x5 + swapInductionVars hoist)

**setAttrs** (5 处, 行 836/980/1199/1449/1732):
```diff
-  op->setAttrs({{"packing", rewriter.getStringAttr("filter")},
-                {"multipacking", rewriter.getBoolAttr(false)}});
+  op->setAttrs(rewriter.getNamedAttr("packing", rewriter.getStringAttr("filter")));
+  op->setAttr("multipacking", rewriter.getBoolAttr(false));
```

**swapInductionVars** (行 1062, SSA 支配修复):
```diff
+  auto hoistIfLocal = [&](Value &v) {
+    if (auto *defOp = v.getDefiningOp()) {
+      if (defOp->getBlock() == innerLoop.getBody() ||
+          defOp->getBlock() == outerLoop.getBody())
+        defOp->moveBefore(outerLoop);
+    }
+  };
+  hoistIfLocal(innerLowerBound);
+  hoistIfLocal(innerUpperBound);
+  hoistIfLocal(innerStep);
+
   outerLoop.setLowerBound(innerLowerBound);
   outerLoop.setUpperBound(innerUpperBound);
   outerLoop.setStep(innerStep);
```

---

## 测试结果

### Transform 测试 (sconv-opt)
18/18 全部通过 (6 transform x 3 payload):

| Transform | payload (128x66->256) | pay2load (96x56->24 1x1) | paysload (332x30->336) |
|-----------|:-:|:-:|:-:|
| sconv.mlir | OK | OK | OK |
| sconv_mK.mlir | OK | OK | OK |
| sconv_arch.mlir | OK | OK | OK |
| sconv_latency.mlir | OK | OK | OK |
| s2conv.mlir | OK | OK | OK |
| s2conv_mK.mlir | OK | OK | OK |

### 端到端 JIT (transform + mlir-opt + mlir-cpu-runner)
- payload: 1x128x66x66 -> 256x128x3x3 -> 1x256x64x64, stride=1, dilation=1
- transform: sconv_mK.mlir (mK_info=[16,8])
- **Baseline vs SConv transformed: 数值逐元素一致** (输出最后一行
  `[4.90186e+07, ..., 7.22795e+07]` 完全相同)
