$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path -Path $DesktopPath -ChildPath "TRELLIS 3D Studio.lnk"
$TargetPath = Join-Path -Path $PSScriptRoot -ChildPath "dist\TRELLIS_3D_Studio.exe"
$IconPath = Join-Path -Path $PSScriptRoot -ChildPath "output\app_logo.ico"

if (-not (Test-Path $TargetPath)) {
    $TargetPath = Join-Path -Path $PSScriptRoot -ChildPath "TRELLIS_3D_Studio.exe"
}

$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetPath
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.Description = "TRELLIS 3D Studio - Standalone Desktop Application"
$Shortcut.IconLocation = "$IconPath, 0"
$Shortcut.Save()

Write-Host "Created Desktop Shortcut: $ShortcutPath"
