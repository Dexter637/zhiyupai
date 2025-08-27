# 设置Docker国内镜像源脚本
# 管理员身份运行此脚本

Write-Host "正在配置Docker国内镜像源..."

# 创建或修改Docker配置文件
$dockerConfigPath = "$env:ProgramData\Docker\config\daemon.json"

# 检查配置文件是否存在
if (Test-Path $dockerConfigPath) {
    Write-Host "检测到已存在的daemon.json配置文件，将进行备份..."
    Copy-Item $dockerConfigPath "$dockerConfigPath.bak_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}

# 创建新的配置内容
$configContent = @"
{
  "registry-mirrors": [
    "https://mirrors.cn99.com",
    "https://docker.mirrors.ustc.edu.cn",
    "https://registry.docker-cn.com",
    "https://hub-mirror.c.163.com"
  ]
}
"@

# 写入配置文件
try {
    New-Item -ItemType File -Path $dockerConfigPath -Force | Out-Null
    Set-Content -Path $dockerConfigPath -Value $configContent -Encoding UTF8
    Write-Host "Docker镜像源配置成功！"
    
    # 显示配置内容
    Write-Host "\n当前Docker镜像源配置："
    Get-Content $dockerConfigPath
    
    Write-Host "\n请重启Docker服务使配置生效。"
    Write-Host "您可以通过以下命令重启Docker服务："
    Write-Host "Restart-Service docker"
    Write-Host "\n配置的镜像源列表："
    Write-Host "1. https://mirrors.cn99.com (您提供的镜像源)"
    Write-Host "2. https://docker.mirrors.ustc.edu.cn (中科大镜像)"
    Write-Host "3. https://registry.docker-cn.com (Docker中国官方镜像)"
    Write-Host "4. https://hub-mirror.c.163.com (网易镜像)"
} catch {
    Write-Host "配置失败：$($_.Exception.Message)"
    Write-Host "请以管理员身份运行此脚本。"
}

Write-Host "\n脚本执行完成。"