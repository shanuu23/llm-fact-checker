-- Resources may be translocated by macOS. Never rely on files beside the app.
on run
    try
        set helperPath to POSIX path of (path to resource "macos_workspace.sh")
        set resourcesPath to POSIX path of (path to resource "release-id.txt")
        set supportPath to (POSIX path of (path to application support from user domain)) & "ModelLens"
        set workspacePath to do shell script "/bin/bash " & quoted form of helperPath & " " & quoted form of resourcesPath & " " & quoted form of supportPath
        set launchCommand to "exec /usr/bin/env MODELLENS_DESKTOP=1 /bin/bash " & quoted form of (workspacePath & "/Launch ModelLens.command")
        tell application "Terminal"
            activate
            do script launchCommand
        end tell
    on error errorText number errorNumber
        if errorNumber is -1743 then
            set errorText to "ModelLens needs permission to open its setup window in Terminal. Review System Settings > Privacy & Security > Automation, allow ModelLens to control Terminal if you trust this copy, then reopen ModelLens."
        end if
        display dialog errorText with title "ModelLens could not start" buttons {"OK"} default button "OK" with icon caution
    end try
end run
