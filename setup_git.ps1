# Git Setup and Push Script
# This script initializes the git repository and pushes to GitHub

Write-Host "=== Kazakhstan House Price Predictor - Git Setup ===" -ForegroundColor Cyan

# Check if git is installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Git is not installed. Please install Git first." -ForegroundColor Red
    exit 1
}

# Navigate to project directory
$projectPath = "c:\Users\00055794\Desktop\Gulnaz\PROJECTS\House_Prices\102025_3"
Set-Location $projectPath

Write-Host "`nCurrent directory: $projectPath" -ForegroundColor Green

# Check if .git exists
if (Test-Path ".git") {
    Write-Host "`nGit repository already exists." -ForegroundColor Yellow
    $reinit = Read-Host "Do you want to reinitialize? (yes/no)"
    if ($reinit -eq "yes") {
        Remove-Item -Recurse -Force ".git"
        Write-Host "Removed existing .git directory" -ForegroundColor Yellow
    } else {
        Write-Host "Keeping existing repository" -ForegroundColor Green
    }
}

# Initialize git if needed
if (-not (Test-Path ".git")) {
    Write-Host "`nInitializing git repository..." -ForegroundColor Green
    git init
    git branch -M main
}

# Check current status
Write-Host "`nChecking repository status..." -ForegroundColor Green
git status

# Show what will be ignored
Write-Host "`nFiles that will be ignored by .gitignore:" -ForegroundColor Cyan
Write-Host "  - .venv/ (virtual environment - ~1.2GB)" -ForegroundColor Gray
Write-Host "  - __pycache__/ (Python cache files)" -ForegroundColor Gray
Write-Host "  - .ipynb_checkpoints (Jupyter checkpoints)" -ForegroundColor Gray
Write-Host "  - IDE configuration files" -ForegroundColor Gray

# Show project size
Write-Host "`nProject size (excluding .venv):" -ForegroundColor Cyan
$size = (Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notlike "*\.venv\*" } | Measure-Object -Property Length -Sum).Sum
$sizeMB = [math]::Round($size / 1MB, 2)
Write-Host "  $sizeMB MB" -ForegroundColor Green

# Add files
Write-Host "`nAdding files to git..." -ForegroundColor Green
git add .

# Show what's staged
Write-Host "`nStaged files:" -ForegroundColor Green
git status --short

# Commit
$commitMsg = "Initial commit - optimized for deployment (size: $sizeMB MB)"
Write-Host "`nCommitting with message: $commitMsg" -ForegroundColor Green
git commit -m $commitMsg

# Remote setup
Write-Host "`n=== GitHub Remote Setup ===" -ForegroundColor Cyan
Write-Host "Repository: https://github.com/andprov/krisha.kz.git" -ForegroundColor Yellow

$setupRemote = Read-Host "`nDo you want to add/update the remote origin? (yes/no)"
if ($setupRemote -eq "yes") {
    # Check if remote exists
    $remoteExists = git remote get-url origin 2>$null
    if ($remoteExists) {
        Write-Host "Updating existing remote..." -ForegroundColor Yellow
        git remote set-url origin https://github.com/andprov/krisha.kz.git
    } else {
        Write-Host "Adding new remote..." -ForegroundColor Green
        git remote add origin https://github.com/andprov/krisha.kz.git
    }
    
    Write-Host "`nRemote configured successfully!" -ForegroundColor Green
    
    $pushNow = Read-Host "`nDo you want to push to GitHub now? (yes/no)"
    if ($pushNow -eq "yes") {
        Write-Host "`nPushing to GitHub..." -ForegroundColor Green
        git push -u origin main
        Write-Host "`nSuccessfully pushed to GitHub!" -ForegroundColor Green
    } else {
        Write-Host "`nTo push later, run: git push -u origin main" -ForegroundColor Yellow
    }
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Green
Write-Host "1. Visit https://github.com/andprov/krisha.kz to verify" -ForegroundColor White
Write-Host "2. Deploy to Render.com using the repository" -ForegroundColor White
Write-Host "3. See DEPLOYMENT.md for detailed instructions" -ForegroundColor White
