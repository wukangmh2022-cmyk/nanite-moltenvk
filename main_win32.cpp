#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

#include <Win32NativeWindow.h>

#include "DemoRenderer.hpp"

namespace
{

constexpr int RabbitEditId = 1001;
constexpr int RabbitSliderId = 1002;
constexpr int ReloadButtonId = 1003;
constexpr int HzbCheckboxId = 1004;
constexpr std::uint32_t MinRabbits = 1;
constexpr std::uint32_t MaxRabbits = 10000;

struct AppState
{
    HWND Window = nullptr;
    HWND RabbitEdit = nullptr;
    HWND RabbitSlider = nullptr;
    std::unique_ptr<NaniteDemo::Renderer> Renderer;
    bool MoveForward = false;
    bool MoveBackward = false;
    bool MoveLeft = false;
    bool MoveRight = false;
    bool Dragging = false;
    POINT LastMouse{};
    float CameraX = 0.0f;
    float CameraY = 0.0f;
    float CameraZ = 6.0f;
    float CameraYaw = 0.0f;
    float CameraPitch = 0.0f;
    float ForwardVelocity = 0.0f;
    float StrafeVelocity = 0.0f;
    LARGE_INTEGER LastTick{};
    LARGE_INTEGER Frequency{};
    std::deque<double> FrameDurations;
    double FrameDurationSum = 0.0;
};

AppState* GetState(HWND window)
{
    return reinterpret_cast<AppState*>(GetWindowLongPtrW(window, GWLP_USERDATA));
}

void SetRabbitEdit(AppState& state, std::uint32_t count)
{
    wchar_t value[32];
    swprintf_s(value, L"%u", count);
    SetWindowTextW(state.RabbitEdit, value);
    SendMessageW(state.RabbitSlider, TBM_SETPOS, TRUE, static_cast<LPARAM>(count));
}

std::uint32_t ReadRabbitEdit(const AppState& state)
{
    wchar_t value[32] = {};
    GetWindowTextW(state.RabbitEdit, value, static_cast<int>(std::size(value)));
    const unsigned long parsed = wcstoul(value, nullptr, 10);
    return static_cast<std::uint32_t>(std::clamp<unsigned long>(
        parsed, MinRabbits, MaxRabbits));
}

void ReloadScene(AppState& state)
{
    const std::uint32_t count = ReadRabbitEdit(state);
    SetRabbitEdit(state, count);
    try
    {
        state.Renderer->ReloadRabbitCount(count);
    }
    catch (const std::exception& error)
    {
        MessageBoxA(state.Window, error.what(), "Nanite scene reload failed", MB_ICONERROR);
    }
}

void LayoutControls(AppState& state, int width)
{
    const int panelWidth = std::min(width, 520);
    MoveWindow(state.RabbitEdit, 78, 12, 72, 26, TRUE);
    MoveWindow(state.RabbitSlider, 164, 12, std::max(panelWidth - 320, 120), 26, TRUE);
    MoveWindow(state.Window == nullptr ? nullptr : GetDlgItem(state.Window, ReloadButtonId),
               panelWidth - 140, 10, 128, 30, TRUE);
}

void UpdateCamera(AppState& state, double deltaTime)
{
    const float forwardInput = (state.MoveForward ? 1.0f : 0.0f) -
        (state.MoveBackward ? 1.0f : 0.0f);
    const float strafeInput = (state.MoveRight ? 1.0f : 0.0f) -
        (state.MoveLeft ? 1.0f : 0.0f);
    const float response = 1.0f - std::exp(-12.0f * static_cast<float>(deltaTime));
    constexpr float moveSpeed = 4.0f;
    state.ForwardVelocity += (forwardInput * moveSpeed - state.ForwardVelocity) * response;
    state.StrafeVelocity += (strafeInput * moveSpeed - state.StrafeVelocity) * response;

    const float forwardX = std::sin(state.CameraYaw);
    const float forwardZ = -std::cos(state.CameraYaw);
    const float rightX = std::cos(state.CameraYaw);
    const float rightZ = std::sin(state.CameraYaw);
    state.CameraX += (forwardX * state.ForwardVelocity + rightX * state.StrafeVelocity) *
        static_cast<float>(deltaTime);
    state.CameraZ += (forwardZ * state.ForwardVelocity + rightZ * state.StrafeVelocity) *
        static_cast<float>(deltaTime);
}

void UpdateTitle(AppState& state)
{
    const Nanite::FrameStats& stats = state.Renderer->GetLastFrameStats();
    const double fps = state.FrameDurationSum > 0.0 ?
        static_cast<double>(state.FrameDurations.size()) / state.FrameDurationSum : 0.0;
    char title[256];
    std::snprintf(
        title,
        sizeof(title),
        "Nanite | %.1f FPS | Rabbits %u | Tri %.3fM | I%u/%u C%u/%u | H%u | HZB %s",
        fps,
        stats.InstanceCount,
        static_cast<double>(stats.LogicalTriangleCount) / 1000000.0,
        stats.InstanceCount,
        stats.VisibleInstanceCount,
        stats.ClusterCandidateCount,
        stats.DrawCount,
        stats.ClusterHZBRejected,
        stats.HZBEnabled ? "ON" : "OFF");
    SetWindowTextA(state.Window, title);
}

void Tick(AppState& state)
{
    LARGE_INTEGER now{};
    QueryPerformanceCounter(&now);
    const double deltaTime = state.LastTick.QuadPart == 0 ? 0.0 :
        static_cast<double>(now.QuadPart - state.LastTick.QuadPart) /
            static_cast<double>(state.Frequency.QuadPart);
    state.LastTick = now;
    const double clampedDelta = std::min(deltaTime, 0.1);
    UpdateCamera(state, clampedDelta);
    state.Renderer->SetCamera(
        state.CameraX, state.CameraY, state.CameraZ, state.CameraYaw, state.CameraPitch);

    RECT client{};
    GetClientRect(state.Window, &client);
    state.Renderer->Draw(
        static_cast<std::uint32_t>(std::max(0L, client.right - client.left)),
        static_cast<std::uint32_t>(std::max(0L, client.bottom - client.top)));

    if (deltaTime > 0.0)
    {
        state.FrameDurations.push_back(deltaTime);
        state.FrameDurationSum += deltaTime;
        while (state.FrameDurations.size() > 240)
        {
            state.FrameDurationSum -= state.FrameDurations.front();
            state.FrameDurations.pop_front();
        }
    }
    UpdateTitle(state);
}

void SetMovementKey(AppState& state, WPARAM key, bool pressed)
{
    switch (key)
    {
        case 'W': case VK_UP: state.MoveForward = pressed; break;
        case 'S': case VK_DOWN: state.MoveBackward = pressed; break;
        case 'A': case VK_LEFT: state.MoveLeft = pressed; break;
        case 'D': case VK_RIGHT: state.MoveRight = pressed; break;
        default: break;
    }
}

LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    AppState* state = GetState(window);
    switch (message)
    {
        case WM_NCCREATE:
        {
            auto* create = reinterpret_cast<CREATESTRUCTW*>(lParam);
            state = static_cast<AppState*>(create->lpCreateParams);
            state->Window = window;
            SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
            return TRUE;
        }
        case WM_CREATE:
        {
            InitCommonControls();
            state->RabbitEdit = CreateWindowExW(
                WS_EX_CLIENTEDGE, L"EDIT", L"27",
                WS_CHILD | WS_VISIBLE | ES_NUMBER | ES_AUTOHSCROLL,
                78, 12, 72, 26, window, reinterpret_cast<HMENU>(RabbitEditId),
                GetModuleHandleW(nullptr), nullptr);
            state->RabbitSlider = CreateWindowExW(
                0, TRACKBAR_CLASSW, L"",
                WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS,
                164, 12, 220, 26, window, reinterpret_cast<HMENU>(RabbitSliderId),
                GetModuleHandleW(nullptr), nullptr);
            SendMessageW(state->RabbitSlider, TBM_SETRANGE, TRUE, MAKELONG(MinRabbits, MaxRabbits));
            SendMessageW(state->RabbitSlider, TBM_SETPOS, TRUE, 27);
            CreateWindowExW(
                0, L"BUTTON", L"HZB enabled",
                WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                12, 44, 140, 24, window, reinterpret_cast<HMENU>(HzbCheckboxId),
                GetModuleHandleW(nullptr), nullptr);
            CreateWindowExW(
                0, L"BUTTON", L"Reload Scene",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                392, 10, 128, 30, window, reinterpret_cast<HMENU>(ReloadButtonId),
                GetModuleHandleW(nullptr), nullptr);

            try
            {
                Diligent::NativeWindow nativeWindow{window};
                state->Renderer = std::make_unique<NaniteDemo::Renderer>(nativeWindow, 1280, 720);
                state->Renderer->SetHZBEnabled(false);
                SetRabbitEdit(*state, state->Renderer->GetRabbitCount());
            }
            catch (const std::exception& error)
            {
                MessageBoxA(window, error.what(), "Nanite startup failed", MB_ICONERROR);
                PostMessageW(window, WM_CLOSE, 0, 0);
                return -1;
            }
            QueryPerformanceFrequency(&state->Frequency);
            SetTimer(window, 1, 16, nullptr);
            return 0;
        }
        case WM_COMMAND:
            if (LOWORD(wParam) == ReloadButtonId && HIWORD(wParam) == BN_CLICKED)
                ReloadScene(*state);
            else if (LOWORD(wParam) == HzbCheckboxId && HIWORD(wParam) == BN_CLICKED)
            {
                const bool enabled = SendMessageW(
                    GetDlgItem(window, HzbCheckboxId), BM_GETCHECK, 0, 0) == BST_CHECKED;
                state->Renderer->SetHZBEnabled(enabled);
            }
            return 0;
        case WM_HSCROLL:
            if (reinterpret_cast<HWND>(lParam) == state->RabbitSlider)
                SetRabbitEdit(*state, static_cast<std::uint32_t>(SendMessageW(
                    state->RabbitSlider, TBM_GETPOS, 0, 0)));
            return 0;
        case WM_SIZE:
            if (state != nullptr)
                LayoutControls(*state, LOWORD(lParam));
            return 0;
        case WM_KEYDOWN:
            SetMovementKey(*state, wParam, true);
            return 0;
        case WM_KEYUP:
            SetMovementKey(*state, wParam, false);
            return 0;
        case WM_LBUTTONDOWN:
            state->Dragging = true;
            state->LastMouse = {GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
            SetCapture(window);
            return 0;
        case WM_MOUSEMOVE:
            if (state->Dragging)
            {
                const POINT current{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
                state->CameraYaw += static_cast<float>(current.x - state->LastMouse.x) * 0.006f;
                state->CameraPitch = std::clamp(
                    state->CameraPitch - static_cast<float>(current.y - state->LastMouse.y) * 0.006f,
                    -1.45f, 1.45f);
                state->LastMouse = current;
            }
            return 0;
        case WM_LBUTTONUP:
            state->Dragging = false;
            ReleaseCapture();
            return 0;
        case WM_TIMER:
            if (wParam == 1 && state->Renderer != nullptr)
                Tick(*state);
            return 0;
        case WM_ERASEBKGND:
            return 1;
        case WM_DESTROY:
            KillTimer(window, 1);
            state->Renderer.reset();
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(window, message, wParam, lParam);
    }
}

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int showCommand)
{
    SetProcessDPIAware();
    const wchar_t className[] = L"DiligentCoreNaniteDemo";
    WNDCLASSW windowClass{};
    windowClass.hInstance = instance;
    windowClass.lpfnWndProc = WindowProc;
    windowClass.lpszClassName = className;
    windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    windowClass.hbrBackground = nullptr;
    if (RegisterClassW(&windowClass) == 0)
        return 1;

    AppState state;
    HWND window = CreateWindowExW(
        0,
        className,
        L"DiligentCore Nanite Demo",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        1280,
        760,
        nullptr,
        nullptr,
        instance,
        &state);
    if (window == nullptr)
        return 1;

    ShowWindow(window, showCommand);
    UpdateWindow(window);
    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0)
    {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    return static_cast<int>(message.wParam);
}
