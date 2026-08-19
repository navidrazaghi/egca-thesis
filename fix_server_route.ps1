# Keep the training server reachable while the VPN stays connected.
#
# The VPN captures the default route, so ssh to the server goes into the tunnel
# and times out.  A host route for that one address, pointed at the physical
# adapter's gateway, keeps everything else in the tunnel untouched.
#
# The catch is that such a route is tied to the gateway and interface of the
# network it was created on.  Join a different Wi-Fi and both change -- this has
# already happened three times, each time looking like "the server is down".
# So this script reads the current values rather than hard-coding them.
#
# Run in an elevated PowerShell:
#     powershell -ExecutionPolicy Bypass -File fix_server_route.ps1
#
# Note: this deliberately sends traffic to the server outside the VPN, using the
# real IP of this machine.

param([string]$Server = "213.233.184.253")

$ErrorActionPreference = "Stop"

$admin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host "This needs an elevated PowerShell (Run as Administrator)." `
        -ForegroundColor Red
    exit 1
}

# The physical adapter is the one with a default gateway that is not the VPN's.
# TAP/TUN adapters name themselves after the client, so they are matched by
# description rather than by alias, which the user can rename.
$candidates = Get-NetIPConfiguration |
    Where-Object {
        $_.NetAdapter.Status -eq "Up" -and
        $_.IPv4DefaultGateway -and
        $_.NetAdapter.InterfaceDescription -notmatch "TAP|TUN|VPN|WireGuard|Hyper-V|Virtual"
    }

if (-not $candidates) {
    Write-Host "No physical adapter with a gateway found. Is the Wi-Fi up?" `
        -ForegroundColor Red
    exit 1
}

# Lowest interface metric wins, matching how Windows itself would choose.
$nic = $candidates | Sort-Object { $_.NetAdapter.ifIndex } | Select-Object -First 1
$gw  = $nic.IPv4DefaultGateway.NextHop
$idx = $nic.InterfaceIndex

Write-Host "adapter : $($nic.InterfaceAlias) (if $idx)"
Write-Host "gateway : $gw"

$existing = Get-NetRoute -DestinationPrefix "$Server/32" -ErrorAction SilentlyContinue
if ($existing) {
    foreach ($r in $existing) {
        Write-Host "removing stale route via $($r.NextHop) on if $($r.ifIndex)"
    }
    route delete $Server | Out-Null
}

route -p add $Server mask 255.255.255.255 $gw metric 1 if $idx | Out-Null

Write-Host ""
Write-Host "testing ..." -NoNewline
$ok = Test-NetConnection -ComputerName $Server -Port 22 -WarningAction SilentlyContinue
if ($ok.TcpTestSucceeded) {
    Write-Host " reachable on port 22." -ForegroundColor Green
} else {
    Write-Host " still unreachable." -ForegroundColor Yellow
    Write-Host "If the VPN client enforces a kill switch it will override this" `
        "route; exclude $Server in the VPN profile instead."
}
