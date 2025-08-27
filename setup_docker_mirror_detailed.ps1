# Docker镜像源配置详细脚本
# 管理员身份运行此脚本

Write-Host "=== Docker国内镜像源配置工具 ==="
Write-Host "此脚本将帮助您配置多个国内Docker镜像源，加速镜像下载。"
Write-Host "\n请确保已关闭Docker Desktop应用程序。"
Write-Host "\n按任意键继续..."
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')

# 定义镜像源列表
$mirrors = @(
    "https://mirrors.cn99.com",
    "https://docker.mirrors.ustc.edu.cn",
    "https://registry.docker-cn.com",
    "https://hub-mirror.c.163.com",
    "https://mirror.ccs.tencentyun.com",
    "https://docker.mirrors.sjtug.sjtu.edu.cn"
)

# 创建配置文件路径
$dockerConfigDir = "$env:ProgramData\Docker\config"
$dockerConfigPath = "$dockerConfigDir\daemon.json"

# 检查并创建配置目录
if (-not (Test-Path $dockerConfigDir)) {
    Write-Host "创建Docker配置目录: $dockerConfigDir"
    New-Item -ItemType Directory -Path $dockerConfigDir -Force | Out-Null
}

# 检查配置文件是否存在
if (Test-Path $dockerConfigPath) {
    Write-Host "检测到已存在的daemon.json配置文件，将进行备份..."
    $backupPath = "$dockerConfigPath.bak_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Copy-Item $dockerConfigPath $backupPath -Force
    Write-Host "配置文件已备份至: $backupPath"
}

# 创建新的配置内容
$configContent = @"
{
  "registry-mirrors": [
"@

# 添加镜像源
foreach ($mirror in $mirrors) {
    $configContent += "    \"$mirror\",
"
}

# 移除最后一个逗号并关闭数组
$configContent = $configContent.TrimEnd(",\r\n")
$configContent += @"
  ]
}
"@

# 写入配置文件
try {
    New-Item -ItemType File -Path $dockerConfigPath -Force | Out-Null
    Set-Content -Path $dockerConfigPath -Value $configContent -Encoding UTF8
    Write-Host "\nDocker镜像源配置成功！"
    
    # 显示配置内容
    Write-Host "\n当前Docker镜像源配置："
    Get-Content $dockerConfigPath | Write-Host
    
    Write-Host "\n=== 配置完成 ==="
    Write-Host "1. 请手动重启Docker Desktop应用程序使配置生效"
    Write-Host "2. 如果重启后仍然无法下载镜像，请尝试以下操作："
    Write-Host "   - 检查网络连接是否正常"
    Write-Host "   - 尝试暂时关闭防火墙或安全软件"
    Write-Host "   - 在Docker Desktop设置中再次确认镜像源配置"
    Write-Host "\n配置的镜像源列表："
    for ($i = 0; $i -lt $mirrors.Length; $i++) {
        Write-Host "  $($i+1). $($mirrors[$i])"
    }
} catch {
    Write-Host "\n配置失败：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host "请以管理员身份运行此脚本。" -ForegroundColor Red
}

Write-Host "\n脚本执行完成。按任意键退出。"
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')