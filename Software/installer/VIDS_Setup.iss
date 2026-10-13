; VIDS Setup - Inno Setup 6 script. Compile with ISCC.exe (no admin needed).
; Per-user install into %LOCALAPPDATA%\VIDS (works on college PCs).

#define MyAppName "Voice Impersonation Detection System"
#define MyAppShort "VIDS"
#define MyAppVersion "1.0"
#define MyAppPublisher "KISHORE G K (24ADU297)"

[Setup]
AppId={{8F3A1B2C-VIDS-4D5E-9F01-PROFESSORK}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\VIDS
OutputDir=..
OutputBaseFilename=Setup
Compression=lzma2/fast
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
UninstallDisplayName={#MyAppShort} - Voice Impersonation Detection System
WizardStyle=modern
DisableProgramGroupPage=yes
SetupIconFile=..\logo.ico
WizardImageFile=wiz_image.bmp
WizardSmallImageFile=wiz_small.bmp

[Files]
Source: "..\dist\VIDS\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppShort}"; Filename: "{app}\VIDS.exe"
Name: "{autodesktop}\{#MyAppShort}"; Filename: "{app}\VIDS.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\VIDS.exe"; Description: "Launch {#MyAppShort} now"; Flags: nowait postinstall skipifsilent
