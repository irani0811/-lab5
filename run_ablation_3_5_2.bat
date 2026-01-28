@echo off
setlocal enabledelayedexpansion

set "FUSION=co_attn"
set "BASE_PREFIX=outputs_352"

set "EPOCHS=6"
set "SEED=42"
set "VAL_RATIO=0.2"
set "BATCH=16"

for %%C in (0 1) do (
  set "CLEAN="
  if %%C==1 set "CLEAN=--clean-text"

  for %%A in (none weak strong) do (
    set "AUG="
    if "%%A"=="weak" set "AUG=--use-image-aug"
    if "%%A"=="strong" set "AUG=--use-image-aug --use-strong-image-aug"

    for %%R in (base ls rdrop md) do (
      set "REG="
      if "%%R"=="ls" set "REG=--label-smoothing 0.1"
      if "%%R"=="rdrop" set "REG=--rdrop-alpha 0.5"
      if "%%R"=="md" set "REG=--modality-dropout-prob 0.2"

      set "OUT=!BASE_PREFIX!_c%%C_%%A_%%R"

      python train_multimodal.py --mode train --data-dir data --train-file train.txt --epochs !EPOCHS! --seed !SEED! --val-ratio !VAL_RATIO! --batch-size !BATCH! --num-workers 0 --use-amp --fusion-method !FUSION! !CLEAN! !AUG! !REG! --output-dir !OUT! --checkpoint-path !OUT!\best_model.pt
      if errorlevel 1 exit /b 1
    )
  )
)

endlocal
