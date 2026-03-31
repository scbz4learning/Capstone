$vsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

cmd /c "`"$vsDevCmd`" && set" |
ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        $name = $matches[1]
        $value = $matches[2]

        [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}