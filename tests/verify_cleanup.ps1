$keep = @(
  'C:\Users\Kishore GK\Desktop\VIDS\model\professor_k.pth',
  'C:\Users\Kishore GK\Desktop\VIDS\Report\Voice_Impersonation_Detection_System_Report.docx',
  'C:\Users\Kishore GK\Desktop\VIDS\webapp\app.py',
  'C:\Users\Kishore GK\Desktop\VIDS\Software\Setup.exe',
  'C:\Users\Kishore GK\Desktop\Software\Setup.exe',
  'C:\Users\Kishore GK\Desktop\VIDS\dataset\field\manifest.json'
)
'=== keep-list ==='
foreach ($f in $keep) { '{0} : {1}' -f (Test-Path $f), ([System.IO.Path]::GetFileName($f)) }
'=== 6 keepers ==='
$files = @('real_test_1.spk1089.mp3','real_test_2.spk1188.mp3','real_test_4.spk1221.mp3','male ai.mp3','femail ai.mp3','motivation is a myth.mp3')
foreach ($f in $files) { '{0} : {1}' -f (Test-Path "C:\Users\Kishore GK\Downloads\$f"), $f }
'=== deleted confirm ==='
foreach ($d in @('C:\Users\Kishore GK\Desktop\VIDS\dataset\mega','C:\Users\Kishore GK\Desktop\VIDS\dataset\hf_mlaad_tiny','C:\Users\Kishore GK\Desktop\VIDS\dataset\balanced')) { '{0} gone: {1}' -f $d, (-not (Test-Path $d)) }
