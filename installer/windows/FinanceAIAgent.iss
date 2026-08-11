; Inno Setup scaffold for the Windows onedir application.
#define MyAppName "Finance AI Agent"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Finance AI Agent"
#define MyAppExeName "Finance AI Agent.exe"

[Setup]
AppId={{A4C44661-2E65-4F1F-9347-8CA3D5B4D948}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Finance AI Agent
DefaultGroupName=Finance AI Agent
OutputDir=..\..\dist\installer
OutputBaseFilename=Finance-AI-Agent-Setup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Tasks]
Name: "desktopicon"; Description: "Crear un acceso directo en el escritorio"; GroupDescription: "Accesos directos:"; Flags: unchecked

[Files]
Source: "..\..\dist\Finance AI Agent\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Finance AI Agent"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\Finance AI Agent"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Abrir Finance AI Agent y completar la configuración"; Flags: nowait postinstall skipifsilent

; User data lives under %LOCALAPPDATA%\FinanceAIAgent and is deliberately not
; listed in [Files] or [UninstallDelete], so upgrades/uninstalls preserve it.

[Code]
function OllamaInstalled(): Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec(ExpandConstant('{cmd}'), '/C where ollama', '', SW_HIDE,
    ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
  if not Result then
    Result := FileExists(ExpandConstant('{localappdata}\Programs\Ollama\ollama.exe'));
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and not OllamaInstalled() then
    MsgBox('Finance AI Agent requiere Ollama para el análisis con IA. ' +
      'Instálelo desde https://ollama.com/download y luego abra la aplicación. ' +
      'El modelo no se descargará sin su autorización.', mbInformation, MB_OK);
end;
