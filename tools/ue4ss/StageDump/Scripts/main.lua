-- SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
-- SPDX-License-Identifier: GPL-3.0-or-later

-- StageDump (acr-stagegen): list functions/classes that look like session or
-- stage start entry points, to find what the Free Practice menu calls.
local here = debug.getinfo(1, "S").source:sub(2):gsub("\\", "/"):match("(.*/)")
local OUT = here .. "../dump.txt"
local KEYS = { "Session", "FreePractice", "StartRally", "StartStage", "LoadStage", "LoadTrack",
               "SelectTrack", "SelectStage", "Travel", "OpenLevel", "StartRace", "RaceSetup",
               "TrackVariant", "GameMode", "Championship" }

local function wanted(name)
    for _, k in ipairs(KEYS) do
        if string.find(name, k) then return true end
    end
    return false
end

ExecuteWithDelay(60000, function()
    local f = io.open(OUT, "w")
    local n = 0
    ForEachUObject(function(obj)
        local ok, full = pcall(function() return obj:GetFullName() end)
        if ok and full and (string.sub(full, 1, 9) == "Function " or string.sub(full, 1, 6) == "Class " or
                string.find(full, "^BlueprintGeneratedClass ") or string.find(full, "^WidgetBlueprintGeneratedClass "))
                and wanted(full) and not string.find(full, "/Script/Engine%.") then
            f:write(full .. "\n")
            n = n + 1
        end
    end)
    f:close()
    print(string.format("[StageDump] wrote %d entries to %s\n", n, OUT))
end)
print("[StageDump] loaded\n")
