$path = 'd:\working_projects\SIH\cyberCast\nootebooks\CyberCast_Complete.ipynb'
$content = Get-Content $path -Raw

$find = '"df_all = pd.concat(all_dfs, ignore_index=True)\n",
    "df_all = df_all.sort_values(''Timestamp'').reset_index(drop=True)\n",
    "\n",
    "# Free memory\n",
    "del all_dfs\n",
    "gc.collect()\n",'

$replace = '"df_all = pd.concat(all_dfs, ignore_index=True)\n",
    "\n",
    "# Free memory before sorting to avoid MemoryError\n",
    "del all_dfs\n",
    "import gc\n",
    "gc.collect()\n",
    "\n",
    "df_all.sort_values(''Timestamp'', inplace=True)\n",
    "df_all.reset_index(drop=True, inplace=True)\n",'

$content = $content.Replace($find, $replace)
Set-Content $path -Value $content -Encoding UTF8
