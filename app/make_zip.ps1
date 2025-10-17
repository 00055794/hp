$ErrorActionPreference = "Stop"
$zipName = "kz_house_price_app.zip"
if (Test-Path $zipName) { Remove-Item $zipName }
Compress-Archive -Path .\* -DestinationPath $zipName -Force
Write-Host "Created $zipName"