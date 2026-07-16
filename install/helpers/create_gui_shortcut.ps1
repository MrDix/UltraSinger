# Creates Desktop and Start Menu shortcuts for the UltraSinger GUI.
#
# - Icon: src\gui\resources\icons\logo.ico (the UltraSinger logo)
# - Target: cmd.exe /c run_gui_on_windows.bat - shortcuts whose target is a
#   batch file cannot be pinned to the taskbar, a cmd.exe-hosted one can.
# - AppUserModelID: "UltraSinger.GUI" is stamped onto the shortcut's property
#   store. The GUI process sets the same ID at startup, so the running window
#   groups onto a pinned shortcut (and pinning the running app relaunches it
#   through the Start Menu shortcut) instead of appearing as a second,
#   generic icon.
#
# Run from anywhere; the repo root is derived from this script's location.
# Exit codes: 0 = at least one shortcut created, 1 = nothing created.

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$launcher = Join-Path $repoRoot 'run_gui_on_windows.bat'
$iconPath = Join-Path $repoRoot 'src\gui\resources\icons\logo.ico'
$appId = 'UltraSinger.GUI'  # must match SetCurrentProcessExplicitAppUserModelID in src\gui_main.py

if (-not (Test-Path $launcher)) {
    Write-Host "ERROR: $launcher not found - run this from a complete checkout."
    exit 1
}

# COM interop to stamp System.AppUserModel.ID onto a .lnk. WScript.Shell
# cannot write shell property stores, so this goes through IPropertyStore.
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

namespace UltraSingerShortcut
{
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PropertyKey
    {
        public Guid fmtid;
        public uint pid;
        public PropertyKey(Guid f, uint p) { fmtid = f; pid = p; }
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct PropVariant
    {
        [FieldOffset(0)] public ushort vt;
        [FieldOffset(8)] public IntPtr pointerValue;

        public static PropVariant FromString(string value)
        {
            var pv = new PropVariant();
            pv.vt = 31; // VT_LPWSTR
            pv.pointerValue = Marshal.StringToCoTaskMemUni(value);
            return pv;
        }

        [DllImport("ole32.dll")]
        private static extern int PropVariantClear(ref PropVariant pvar);
        public void Clear() { PropVariantClear(ref this); }
    }

    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"),
     InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IPropertyStore
    {
        void GetCount(out uint cProps);
        void GetAt(uint iProp, out PropertyKey pkey);
        void GetValue(ref PropertyKey key, out PropVariant pv);
        void SetValue(ref PropertyKey key, ref PropVariant pv);
        void Commit();
    }

    public static class Api
    {
        // PKEY_AppUserModel_ID = {9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}, 5
        private static readonly Guid AumidFmtid =
            new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3");
        private static readonly Guid ShellLinkClsid =
            new Guid("00021401-0000-0000-C000-000000000046");

        public static void SetAppId(string lnkPath, string appId)
        {
            object link = Activator.CreateInstance(
                Type.GetTypeFromCLSID(ShellLinkClsid));
            try
            {
                ((IPersistFile)link).Load(lnkPath, 2 /* STGM_READWRITE */);
                var store = (IPropertyStore)link;
                var key = new PropertyKey(AumidFmtid, 5);
                var pv = PropVariant.FromString(appId);
                try { store.SetValue(ref key, ref pv); store.Commit(); }
                finally { pv.Clear(); }
                ((IPersistFile)link).Save(lnkPath, true);
            }
            finally { Marshal.ReleaseComObject(link); }
        }
    }
}
"@

$shell = New-Object -ComObject WScript.Shell
$locations = @(
    @{ Dir = [Environment]::GetFolderPath('Desktop');  Label = 'Desktop' },
    @{ Dir = [Environment]::GetFolderPath('Programs'); Label = 'Start Menu' }
)

$created = 0
foreach ($loc in $locations) {
    if (-not $loc.Dir -or -not (Test-Path $loc.Dir)) { continue }
    $lnkPath = Join-Path $loc.Dir 'UltraSinger.lnk'
    try {
        $lnk = $shell.CreateShortcut($lnkPath)
        $lnk.TargetPath = $env:ComSpec
        # Double-double quotes survive cmd.exe's /c quote stripping, so paths
        # with spaces work.
        $lnk.Arguments = '/c ""' + $launcher + '""'
        $lnk.WorkingDirectory = $repoRoot
        if (Test-Path $iconPath) { $lnk.IconLocation = "$iconPath,0" }
        $lnk.Description = 'UltraSinger GUI'
        $lnk.Save()
        try {
            [UltraSingerShortcut.Api]::SetAppId($lnkPath, $appId)
        } catch {
            # Cosmetic only: without the ID the shortcut still works, the
            # running window just doesn't group onto a pinned copy.
            Write-Host "NOTE: could not tag the $($loc.Label) shortcut with the app ID: $_"
        }
        Write-Host "Created $($loc.Label) shortcut: $lnkPath"
        $created++
    } catch {
        Write-Host "WARNING: could not create the $($loc.Label) shortcut: $_"
    }
}

if ($created -eq 0) {
    Write-Host 'No shortcuts were created.'
    exit 1
}

Write-Host ''
Write-Host 'Taskbar tip: right-click the shortcut (or the running UltraSinger'
Write-Host 'taskbar icon) and choose "Pin to taskbar".'
exit 0
