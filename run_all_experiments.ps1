# run_all_experiments.ps1
#
# Runs both experiments automatically one after another.
# Start this before sleeping, wake up to results.
#
# Usage:
#   .\run_all_experiments.ps1
#
# What it does:
#   1. exp2: Clean PCM + real MUSAN  (baseline)
#      - preprocess -> split -> optuna -> train
#   2. exp3: A-law + real MUSAN      (main contribution)
#      - preprocess -> split -> optuna -> train
#   3. Silero baseline evaluation
#   4. Git commit and push all results
#
# If any step fails, script stops immediately and shows error.
# Check run_log.txt for full output.

$ErrorActionPreference = "Stop"
$LOG = "run_log.txt"
$START = Get-Date

function Log {
    param([string]$msg)
    $timestamp = Get-Date -Format "HH:mm:ss"
    $line = "[$timestamp] $msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

function Run {
    param([string]$script, [string]$desc)
    Log "START: $desc"
    $t = Get-Date
    python $script
    if ($LASTEXITCODE -ne 0) {
        Log "FAILED: $desc (exit code $LASTEXITCODE)"
        Log "Check output above for error details."
        exit 1
    }
    $elapsed = [math]::Round(((Get-Date) - $t).TotalMinutes, 1)
    Log "DONE: $desc ($elapsed min)"
}

function SetExperiment {
    param([string]$exp)
    Log "Switching to experiment: $exp"

    $config = Get-Content "src/training/config.py" -Raw

    # replace the EXPERIMENT line
    $config = $config -replace 'EXPERIMENT = ".*"', "EXPERIMENT = `"$exp`""

    # set fraction to 1.0 for this experiment
    Set-Content "src/training/config.py" $config
    Log "Config updated: EXPERIMENT = $exp, fraction = 1.0"
}

function SetFraction {
    param([string]$exp, [string]$fraction)
    $config = Get-Content "src/training/config.py" -Raw

    # This updates fraction inside the specific experiment block
    # We do a simple sed-like replacement
    $lines = Get-Content "src/training/config.py"
    $inBlock = $false
    $newLines = @()

    foreach ($line in $lines) {
        if ($line -match "`"$exp`"") {
            $inBlock = $true
        }
        if ($inBlock -and $line -match '"fraction"') {
            $line = $line -replace '"fraction":\s+[\d.]+', "`"fraction`":     $fraction"
            $inBlock = $false
        }
        $newLines += $line
    }
    $newLines | Set-Content "src/training/config.py"
    Log "Set fraction=$fraction for $exp"
}

# ── Header ────────────────────────────────────────────────────────────────────
Clear-Content -Path $LOG -ErrorAction SilentlyContinue
Log "================================================="
Log "  I2S VAD Research - Full Training Pipeline"
Log "  Started: $START"
Log "================================================="

# ── Check GPU ─────────────────────────────────────────────────────────────────
Log "Checking GPU..."
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
if ($LASTEXITCODE -ne 0) {
    Log "ERROR: PyTorch not working. Check installation."
    exit 1
}

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Clean PCM + real MUSAN (baseline)
# ══════════════════════════════════════════════════════════════════════════════
Log ""
Log "================================================="
Log "  EXPERIMENT 2: Clean PCM + real MUSAN"
Log "================================================="

SetExperiment "exp2_clean_musan"
SetFraction "exp2_clean_musan" "1.0"

Run "src/data/preprocess.py"    "exp2 preprocess"
Run "src/data/split.py"         "exp2 split"
#Run "src/training/optuna_tune.py" "exp2 optuna tuning"
Run "src/training/train.py"     "exp2 train all models"

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — A-law + real MUSAN (main contribution)
# ══════════════════════════════════════════════════════════════════════════════
Log ""
Log "================================================="
Log "  EXPERIMENT 3: A-law + real MUSAN"
Log "================================================="

SetExperiment "exp3_alaw_musan"
SetFraction "exp3_alaw_musan" "1.0"

Run "src/data/preprocess.py"    "exp3 preprocess"
Run "src/data/split.py"         "exp3 split"
#Run "src/training/optuna_tune.py" "exp3 optuna tuning"
Run "src/training/train.py"     "exp3 train all models"

# ══════════════════════════════════════════════════════════════════════════════
# SILERO BASELINE
# ══════════════════════════════════════════════════════════════════════════════
Log ""
Log "================================================="
Log "  SILERO VAD BASELINE"
Log "================================================="

Run "src/evaluation/silero_baseline.py" "silero baseline evaluation"

# ══════════════════════════════════════════════════════════════════════════════
# GIT COMMIT AND PUSH
# ══════════════════════════════════════════════════════════════════════════════
Log ""
Log "Committing results to GitHub..."
git add outputs/exp2_clean_musan/stats/
git add outputs/exp3_alaw_musan/stats/
git add outputs/silero_baseline/
git add src/
$commitMsg = "full training results exp2 exp3 silero - $(Get-Date -Format 'yyyy-MM-dd')"
git commit -m $commitMsg
git push origin master
Log "Pushed to GitHub."

# ── Final summary ─────────────────────────────────────────────────────────────
$END     = Get-Date
$ELAPSED = [math]::Round(($END - $START).TotalHours, 2)

Log ""
Log "================================================="
Log "  ALL DONE"
Log "  Total time: $ELAPSED hours"
Log "  Finished: $END"
Log "================================================="
Log ""
Log "Results saved:"
Log "  outputs/exp2_clean_musan/stats/final_metrics.csv"
Log "  outputs/exp3_alaw_musan/stats/final_metrics.csv"
Log "  outputs/silero_baseline/silero_results.csv"
Log ""
Log "Next steps:"
Log "  1. Open notebooks/evaluation.ipynb"
Log "  2. Run for exp2, exp3, compare with Silero"
Log "  3. Share CSVs for paper writing"