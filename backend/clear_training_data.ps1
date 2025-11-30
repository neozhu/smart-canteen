# Smart-Canteen 训练数据清理脚本
# 用于清空所有标注数据、训练数据和模型文件

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Smart-Canteen 数据清理工具" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# 定义要清理的路径
$pathsToClean = @(
    @{
        Path = Join-Path $baseDir "data\dataset\images\*"
        Description = "标注图片"
    },
    @{
        Path = Join-Path $baseDir "data\dataset\labels\*"
        Description = "标注标签"
    },
    @{
        Path = Join-Path $baseDir "data\dataset\train"
        Description = "训练集"
        IsDirectory = $true
    },
    @{
        Path = Join-Path $baseDir "data\dataset\val"
        Description = "验证集"
        IsDirectory = $true
    },
    @{
        Path = Join-Path $baseDir "models\best.onnx"
        Description = "ONNX模型"
    },
    @{
        Path = Join-Path $baseDir "models\best.pt"
        Description = "PyTorch模型"
    },
    @{
        Path = Join-Path $baseDir "data\training"
        Description = "训练输出"
        IsDirectory = $true
    }
)

# 统计信息
$totalDeleted = 0
$totalSize = 0

# 清理文件
foreach ($item in $pathsToClean) {
    $path = $item.Path
    $desc = $item.Description
    $isDir = $item.IsDirectory
    
    Write-Host "正在清理: $desc..." -NoNewline
    
    try {
        if ($isDir) {
            # 目录
            if (Test-Path $path) {
                $size = (Get-ChildItem $path -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
                Remove-Item $path -Recurse -Force -ErrorAction Stop
                $totalDeleted++
                $totalSize += $size
                Write-Host " ✓ 已删除 ($([math]::Round($size/1MB, 2)) MB)" -ForegroundColor Green
            } else {
                Write-Host " - 不存在" -ForegroundColor Gray
            }
        } else {
            # 文件或通配符
            $files = Get-Item $path -ErrorAction SilentlyContinue
            if ($files) {
                $count = ($files | Measure-Object).Count
                $size = ($files | Measure-Object -Property Length -Sum).Sum
                Remove-Item $path -Force -ErrorAction Stop
                $totalDeleted += $count
                $totalSize += $size
                Write-Host " ✓ 已删除 $count 个文件 ($([math]::Round($size/1MB, 2)) MB)" -ForegroundColor Green
            } else {
                Write-Host " - 不存在" -ForegroundColor Gray
            }
        }
    } catch {
        Write-Host " ✗ 失败: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "清理完成!" -ForegroundColor Green
Write-Host "共删除: $totalDeleted 项" -ForegroundColor Yellow
Write-Host "释放空间: $([math]::Round($totalSize/1MB, 2)) MB" -ForegroundColor Yellow
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# 显示当前配置
$classesFile = Join-Path $baseDir "data\classes.json"
if (Test-Path $classesFile) {
    try {
        $classes = Get-Content $classesFile -Raw | ConvertFrom-Json
        Write-Host "📝 当前类别配置 ($($classes.Count) 个):" -ForegroundColor Cyan
        foreach ($class in $classes) {
            Write-Host "   - $class" -ForegroundColor White
        }
    } catch {
        Write-Host "⚠️  无法读取类别配置" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  classes.json 不存在" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎯 下一步操作:" -ForegroundColor Green
Write-Host "   1. 确保后端正在运行: python main.py" -ForegroundColor White
Write-Host "   2. 访问标注页面: http://localhost:3000/annotate" -ForegroundColor White
Write-Host "   3. 开始标注数据 (每个类别建议 20-30 张)" -ForegroundColor White
Write-Host "   4. 点击 '🚀 开始训练模型' 按钮" -ForegroundColor White
Write-Host ""
