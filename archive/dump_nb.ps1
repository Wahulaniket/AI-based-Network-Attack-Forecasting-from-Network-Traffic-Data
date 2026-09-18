$json = Get-Content 'nootebooks\CyberCast_Complete.ipynb' -Raw | ConvertFrom-Json
$code = @()
foreach ($cell in $json.cells) {
    if ($cell.cell_type -eq 'code') {
        $code += $cell.source -join ''
        $code += "`n`n"
    }
}
Set-Content 'scratch.py' $code
