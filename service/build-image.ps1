# 构建 job 镜像并自动注入 TorchKiln 的构建身份。
#
# 为什么需要这个脚本（而不是直接 `docker build`）
# ---------------------------------------------
# 「改了 TorchKiln 代码」与「job 镜像是最新的」之间没有任何强制关联，只能靠人记得。
# 2026-10 因此连续踩了两次：
#
#   1. 加了 /api/v1/eval/jobs 端点后没重建 → 评估提交作业直接 404
#   2. 给 MetricSink 加 predict() 后没重建 → 实跑报
#      「AttributeError: 'MetricSink' object has no attribute 'predict'」，
#      而此时 19 张结果图已经写完了（产物齐了、状态却是失败）
#
# 两种报错都指向离真实原因很远的地方。本脚本做两件事：
#
#   * 把当前 git 修订号注入镜像（服务在 /healthz 里回报，平台侧据此判断过期）
#   * 构建完成后打印镜像自报的身份，让「构建成功」这件事本身可核对
#
# 用法（仓库根目录执行）：
#   pwsh service/build-image.ps1                      # 构建默认标签
#   pwsh service/build-image.ps1 -Tag torchkiln:0.1.0
param(
    [string]$Tag = "torchkiln:0.1.0",
    [string]$Dockerfile = "service/Dockerfile"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    # ---- 1. 取构建身份 ----
    $rev = (& git rev-parse --short HEAD 2>$null)
    if (-not $rev) {
        $rev = "nogit"
        Write-Host "  ! 不是 git 仓库或 git 不可用，code_revision 记为 nogit"
    }
    $dirty = $false
    $status = (& git status --porcelain 2>$null)
    if ($status) { $dirty = $true }

    Write-Host "  构建身份: rev=$rev dirty=$dirty"

    # ---- 2. 构建 ----
    $args = @(
        "build", "-t", $Tag,
        "-f", $Dockerfile,
        "--build-arg", "TKILN_REVISION=$rev",
        "--build-arg", "TKILN_DIRTY=$($dirty.ToString().ToLower())",
        "."
    )
    & docker @args
    if ($LASTEXITCODE -ne 0) { throw "docker build 失败（退出码 $LASTEXITCODE）" }

    # ---- 3. 回读镜像自报的身份 ----
    # ⚠️ 两个细节：
    #   * 必须**回读**而不是相信构建参数——ENV 写错、ARG 没透传、基础镜像里残留旧值
    #     这类问题只有回读才发现得了。
    #   * sh 命令必须用 **单引号**传给 PowerShell，否则 `$TKILN_REVISION` 会被
    #     PowerShell 当成自己的变量先展开成空串（已踩过一次，回读得到空值）。
    #   * 这一步是纯 sh，不 import torch，很快。
    $probeCmd = 'echo "rev=$TKILN_REVISION dirty=$TKILN_DIRTY"'
    $out = (& docker run --rm --entrypoint "" $Tag sh -c $probeCmd 2>$null | Select-Object -First 1)
    $revInImage = ""
    $dirtyInImage = ""
    if ($out -match 'rev=(\S*)\s+dirty=(\S*)') {
        $revInImage = $Matches[1]
        $dirtyInImage = $Matches[2]
    }
    Write-Host "  镜像自报: rev=$revInImage dirty=$dirtyInImage"

    if ($revInImage -ne $rev) {
        Write-Host "  ★ 镜像自报的修订号与本地不一致——镜像没带上最新代码。"
        Write-Host "    本地 rev=$rev，镜像 rev=$revInImage"
        exit 1
    }

    # ---- 4. 确认关键能力在镜像里 ----
    # 这些是平台侧会调用的接口/方法。缺任何一个的症状都是运行时的诡异报错
    # （404 / AttributeError），所以在构建时就钉住。
    $probeCmd = @(
        "c=0",
        "grep -q 'api/v1/eval/jobs' /opt/torchkiln/service/main.py || c=1",
        "grep -q 'api/v1/predict/jobs' /opt/torchkiln/service/main.py || c=1",
        "grep -q 'def predict' /opt/torchkiln/ptcore/metrics_sink.py || c=1",
        "grep -q 'build_predict_argv' /opt/torchkiln/service/runner.py || c=1",
        'if [ "$TKILN_REVISION" = "unknown" ]; then echo "rev=unknown" ; fi',
        "exit `$c"
    ) -join "; "
    & docker run --rm --entrypoint "" $Tag sh -c $probeCmd | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ★ 镜像缺少预期的接口/方法（eval/predict 端点、sink.predict、build_predict_argv）"
        Write-Host "    这会导致提交作业时收到 404 或 AttributeError——请重新构建。"
        exit 1
    }
    Write-Host "  接口自检通过: eval 端点 / predict 端点 / sink.predict / build_predict_argv"
    Write-Host "  镜像就绪: $Tag (rev=$revInImage)"
}
finally {
    Pop-Location
}