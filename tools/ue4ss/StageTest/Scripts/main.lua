-- SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
-- SPDX-License-Identifier: GPL-3.0-or-later

-- StageTest (acr-stagegen): camera tour of a generated stage for screenshots.
--  * when the player is in a car, waits ~10 s, switches to the debug camera
--    and moves its SpectatorPawn to each point of
--    ../points.txt ("label x y z yaw", Unreal cm / degrees)
--  * logs "[StageTest] at <label> want=(..) cam=(..)" at each point, "done" at the end
--  * writes the active CommonUI screens (+ IN_CAR) to ../ui.txt for host-side menu automation
--  * live hook: ../live.lua is re-run whenever it changes
-- All game calls run via ExecuteInGameThread. Teleporting the physics car crashes the game.
local here = debug.getinfo(1, "S").source:sub(2):gsub("\\", "/"):match("(.*/)")
local POINTS, LIVE, UI, GO = here .. "../points.txt", here .. "../live.lua", here .. "../ui.txt", here .. "../go.txt"

local function insts(cls)
    local out = {}
    for _, o in ipairs(FindAllOf(cls) or {}) do
        if o:IsValid() and not string.find(o:GetFName():ToString(), "^Default__") then table.insert(out, o) end
    end
    return out
end
local function inst(cls) return insts(cls)[1] end

local function load_points()
    local f = io.open(POINTS, "r")
    if not f then return nil end
    local t = {}
    for line in f:lines() do
        local lab, x, y, z, yaw = line:match("^(%S+)%s+(%S+)%s+(%S+)%s+(%S+)%s+(%S+)")
        if lab then table.insert(t, { lab, tonumber(x), tonumber(y), tonumber(z), tonumber(yaw) }) end
    end
    f:close()
    return t
end

local function controller()
    for _, pc in ipairs(insts("PlayerController")) do
        if pc:GetClass():GetFName():ToString() ~= "DebugCameraController" then return pc end
    end
end

local function console(cmd)
    local pc = controller()
    if pc then StaticFindObject("/Script/Engine.Default__KismetSystemLibrary"):ExecuteConsoleCommand(pc, cmd, pc) end
end

local function in_car()
    local pc = controller()
    local p = pc and pc.Pawn
    if not (p and p:IsValid()) then return false end
    return not string.find(p:GetClass():GetFName():ToString(), "Camera")
end

local function debug_cam_active()            -- the debug camera owns the local player while active
    local d = inst("DebugCameraController")
    if not d then return false end
    local ok, p = pcall(function() return d.Player end)
    return ok and p ~= nil and p:IsValid()
end

local function place(pt)
    -- K2_SetActorLocation (teleport flag) on the game thread moves the debug camera;
    -- K2_TeleportTo silently does nothing here. View from 15 m behind, 6 m up.
    local yaw = math.rad(pt[5])
    local loc = { X = pt[2] - math.cos(yaw) * 1500, Y = pt[3] - math.sin(yaw) * 1500, Z = pt[4] + 600 }
    local sp = FindFirstOf("SpectatorPawn")
    if sp and sp:IsValid() then sp:K2_SetActorLocation(loc, false, {}, true) end
    local d = FindFirstOf("DebugCameraController")
    if d and d:IsValid() then d:SetControlRotation({ Pitch = -15, Yaw = pt[5], Roll = 0 }) end
    return true
end

local function cam_text()
    local d = FindFirstOf("DebugCameraController")
    local l = d and d.PlayerCameraManager:GetCameraLocation()
    return l and string.format("(%.0f, %.0f, %.0f)", l.X / 100, l.Y / 100, l.Z / 100) or "?"
end

-- tour state machine; ticks every 0.5 s, each tick's work runs on the game thread
local state, points, i, t, done = "wait_car", nil, 0, 0, false
local function step()
    if done then return end
    t = t + 1
    if state == "wait_car" then
        -- the start sequence locks the camera: start only once the host says the
        -- car has driven off the line (it writes ../go.txt)
        local g = io.open(GO, "r")
        if g then g:close() end
        if g and in_car() then
            os.remove(GO)
            points = load_points()
            if points and #points > 0 then
                state, t = "settle", 0
                print(string.format("[StageTest] in car, %d points\n", #points))
            end
        else
            t = 0
        end
    elseif state == "settle" and t >= 4 then
        if not debug_cam_active() then console("ToggleDebugCamera") end
        state, t = "cam", 0
    elseif state == "cam" and t >= 4 then               -- debug camera exists after a moment
        state, i, t = "move", 0, 0
    elseif state == "move" then
        i = i + 1
        if i > #points then
            -- the debug camera does not give the car back reliably (toggling,
            -- DisableDebugCamera: no effect), so it stays on; tests restart the game
            print("[StageTest] done\n")
            done = true
            return
        end
        place(points[i])
        state, t = "hold", 0
    elseif state == "hold" and t >= 5 then              -- 2.5 s to stream, then shoot
        print(string.format("[StageTest] at %s want=(%.0f, %.0f) cam=%s\n", points[i][1],
            points[i][2] / 100, points[i][3] / 100, cam_text()))
        state, t = "shot", 0
    elseif state == "shot" and t >= 2 then              -- host grabs within 1 s
        state = "move"
    end
end

local function restart() state, points, i, t, done = "wait_car", nil, 0, 0, false end
StageTestAPI = { console = console, place = place, restart = restart, insts = insts, inst = inst }

-- nothing touches the game in the first seconds: scheduling game-thread
-- callbacks during the logo intro crashed UE4SS natively (only with bigger
-- mod containers mounted, e.g. a generated Elatia)
local T0 = os.time()
local function warm() return os.time() - T0 >= 20 end

LoopAsync(500, function()
    if not warm() then return false end
    ExecuteInGameThread(function()
        local ok, err = pcall(step)
        if not ok then print("[StageTest] error: " .. tostring(err) .. "\n") end
    end)
    return false
end)

LoopAsync(1000, function()
    if not warm() then return false end
    ExecuteInGameThread(function()
        local act = {}
        pcall(function()
            for _, w in ipairs(insts("CommonActivatableWidget")) do
                local ok, a = pcall(function() return w:IsActivated() end)
                if ok and a then act[w:GetClass():GetFName():ToString()] = true end
            end
        end)
        local l = {}
        for k in pairs(act) do table.insert(l, k) end
        table.sort(l)
        local okc, car = pcall(in_car)
        local f = io.open(UI, "w")
        if f then f:write(table.concat(l, ",") .. ((okc and car) and ",IN_CAR" or "")); f:close() end
    end)
    return false
end)

local live_src = nil
LoopAsync(1000, function()
    if not warm() then return false end
    local f = io.open(LIVE, "r")
    if f then
        local src = f:read("*a"); f:close()
        if live_src == nil then live_src = src end          -- don't re-run a stale file at startup
        if src ~= live_src then
            live_src = src
            ExecuteInGameThread(function()
                local chunk, e = load(src, "live")
                local ok, r = false, e
                if chunk then ok, r = pcall(chunk) end
                print("[StageTest] live: " .. (ok and "ok" or tostring(r)) .. "\n")
            end)
        end
    end
    return false
end)
print("[StageTest] loaded\n")
