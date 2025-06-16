@echo off
echo =====================================================
echo  Google Drive の同期を開始します。
echo =====================================================
setlocal

:: ここを編集してください
:: =================================
:: 1. Google Driveのドライブレターを設定
set DRIVE_LETTER=G:

:: 2. 同期したいフォルダのペアをコマンドで指定
:: robocopy "ローカルのフォルダ" "%DRIVE_LETTER%\マイドライブ\Drive上のフォルダ" /MIR
:: =================================

echo [1/3] src フォルダを同期中...
robocopy "C:\Users\kamim\code\Tennis-Analyzer\src" "%DRIVE_LETTER%\マイドライブ\ColabNotebooks\TennisAnalyzer\src" /MIR /NFL /NDL /NJH /NJS /nc /ns /np

echo.
echo [2/3] scripts フォルダを同期中...
robocopy "C:\Users\kamim\code\Tennis-Analyzer\scripts" "%DRIVE_LETTER%\マイドライブ\ColabNotebooks\TennisAnalyzer\scripts" /MIR /NFL /NDL /NJH /NJS /nc /ns /np

echo.
echo [3/3] configs フォルダを同期中...
robocopy "C:\Users\kamim\code\Tennis-Analyzer\configs" "%DRIVE_LETTER%\マイドライブ\ColabNotebooks\TennisAnalyzer\configs" /MIR /NFL /NDL /NJH /NJS /nc /ns /np


echo.
echo =====================================================
echo  すべての同期作業が完了しました。
echo =====================================================
endlocal
pause