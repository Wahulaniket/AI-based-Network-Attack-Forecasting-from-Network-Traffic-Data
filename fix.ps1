$path = 'd:\working_projects\SIH\cyberCast\nootebooks\CyberCast_Complete.ipynb'
$content = Get-Content $path -Raw
$find = '"PROJECT_DIR   = ''/content/drive/MyDrive/CyberCast''\n",'
$replace = '"if IN_COLAB:\n",
    "    PROJECT_DIR   = ''/content/drive/MyDrive/CyberCast''\n",
    "else:\n",
    "    import os\n",
    "    PROJECT_DIR   = os.path.abspath(os.path.join(os.getcwd(), ''..'')) if ''nootebooks'' in os.getcwd() or ''notebooks'' in os.getcwd() else os.getcwd()\n",'
$content = $content.Replace($find, $replace)
Set-Content $path -Value $content -Encoding UTF8
