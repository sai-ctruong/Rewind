# Huong Dan Chay AIC 2026 Rewind

## 1. Chay test

```powershell
cd C:\Users\ad\Downloads\codepython\project\KISC_module
.venv\Scripts\python.exe -m pytest -q
```

## 2. Build index tu du lieu AIC

```powershell
.venv\Scripts\python.exe -m aic2026.cli --data-root data --cache-dir artifacts\aic2026_index --rebuild build-index
```

## 3. Mo giao dien

```powershell
.venv\Scripts\python.exe -m ui.app
```

Mo trinh duyet tai `http://127.0.0.1:5000`.

## 4. Tat server

Neu chay trong terminal, bam `Ctrl+C`.

Neu server chay nen:

```powershell
Get-NetTCPConnection -LocalPort 5000 -State Listen
Stop-Process -Id <OwningProcess> -Force
```

## 5. Xuat submission

Trong UI, chay KIS/Q&A/TRAKE roi bam `Save CSV`. File nam trong `artifacts/submissions/`.