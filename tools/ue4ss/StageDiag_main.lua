-- SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
-- SPDX-License-Identifier: GPL-3.0-or-later

-- StageDiag: log what the engine thinks about ACR track tiles (acr-stagegen diagnostics)
local function vec(v)
    if v == nil then return "nil" end
    return string.format("(%.0f, %.0f, %.0f)", v.X / 100, v.Y / 100, v.Z / 100)
end

local function safe(f)
    local ok, r = pcall(f)
    if ok then return tostring(r) end
    return "err:" .. tostring(r)
end

local dumps = 0
local WANT = { ["0501"] = true, ["0500"] = true, ["0705"] = true }

local function comp_line(c)
    return string.format("[StageDiag]    %s loc=%s visible=%s hiddenInGame=%s registered=%s mainPass=%s mesh=%s class=%s\n",
        safe(function() return c:GetFName():ToString() end),
        safe(function() return vec(c:K2_GetComponentLocation()) end),
        safe(function() return c.bVisible end),
        safe(function() return c.bHiddenInGame end),
        safe(function() return c:IsRegistered() end),
        safe(function() return c.bRenderInMainPass end),
        safe(function() return c.StaticMesh:GetFName():ToString() end),
        safe(function() return c:GetClass():GetFName():ToString() end))
end

local function dump()
    local actors = FindAllOf("Actor")
    if not actors then return end
    local out, n = {}, 0
    for _, a in ipairs(actors) do
        local ok, name = pcall(function() return a:GetFName():ToString() end)
        if ok and type(name) == "string" then
            local key = string.match(name, "Track(%d%d%d%d)_GEN")
            if key then
                n = n + 1
                if WANT[key] then
                    table.insert(out, string.format("[StageDiag] tile %s actor=%s\n", key,
                        safe(function() return vec(a:K2_GetActorLocation()) end)))
                    local root = a.RootComponent
                    table.insert(out, "[StageDiag]  root: " .. comp_line(root))
                    pcall(function()
                        root.AttachChildren:ForEach(function(i, e)
                            table.insert(out, comp_line(e:get()))
                        end)
                    end)
                end
            end
        end
    end
    if n == 0 then return end
    dumps = dumps + 1
    print(string.format("[StageDiag] dump %d: %d tile actors\n", dumps, n) .. table.concat(out))
end

LoopAsync(10000, function()
    local ok, err = pcall(dump)
    if not ok then print("[StageDiag] error: " .. tostring(err) .. "\n") end
    return dumps >= 2
end)
print("[StageDiag] loaded\n")
