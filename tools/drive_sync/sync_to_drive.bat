@echo off
echo =====================================================
echo  Google Drive �̓������J�n���܂��B
echo =====================================================
setlocal

:: ������ҏW���Ă�������
:: =================================
:: 1. Google Drive�̃h���C�u���^�[��ݒ�
set DRIVE_LETTER=G:

:: 2. �����������t�H���_�̃y�A���R�}���h�Ŏw��
:: robocopy "���[�J���̃t�H���_" "%DRIVE_LETTER%\�}�C�h���C�u\Drive��̃t�H���_" /MIR
:: =================================

echo [1/3] src �t�H���_�𓯊���...
robocopy "C:\Users\kamim\code\Tennis-Analyzer\src" "%DRIVE_LETTER%\�}�C�h���C�u\ColabNotebooks\TennisAnalyzer\src" /MIR /NFL /NDL /NJH /NJS /nc /ns /np

echo.
echo [2/3] scripts �t�H���_�𓯊���...
robocopy "C:\Users\kamim\code\Tennis-Analyzer\scripts" "%DRIVE_LETTER%\�}�C�h���C�u\ColabNotebooks\TennisAnalyzer\scripts" /MIR /NFL /NDL /NJH /NJS /nc /ns /np

echo.
echo [3/3] configs �t�H���_�𓯊���...
robocopy "C:\Users\kamim\code\Tennis-Analyzer\configs" "%DRIVE_LETTER%\�}�C�h���C�u\ColabNotebooks\TennisAnalyzer\configs" /MIR /NFL /NDL /NJH /NJS /nc /ns /np


echo.
echo =====================================================
echo  ���ׂĂ̓�����Ƃ��������܂����B
echo =====================================================
endlocal
pause