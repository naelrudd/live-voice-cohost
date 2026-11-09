"""Interactive setup wizard for live-voice-cohost.

Supports both:
  - GUI (Tkinter window) when a display is present (default on Windows desktop)
  - CLI (interactive terminal) with --cli or when Tkinter fails / headless

Runs standalone via:
  setup.bat
  python -m jarvis.setup
  python -m jarvis.setup --cli
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"

PRESETS = {
    "1": {
        "name": "Groq (free tier — recommended)",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-20b,qwen/qwen3.8-27b",
        "key_hint": "Ambil key gratis di https://console.groq.com/keys",
    },
    "2": {
        "name": "OpenRouter (free tier)",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "liquid/lfm-2.5-2.6b:free",
        "key_hint": "Daftar gratis di https://openrouter.ai/keys (no credit card)",
    },
    "3": {
        "name": "Local LLM (Ollama / LM Studio / vLLM)",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:7b",
        "key_hint": "Bisa kosong jika server lokal tidak butuh auth",
    },
    "4": {
        "name": "Custom OpenAI-compatible endpoint",
        "base_url": "",
        "model": "",
        "key_hint": "Masukkan endpoint dan model pilihan Anda",
    },
}


def load_current_env() -> dict[str, str]:
    data: dict[str, str] = {}
    source = ENV_PATH if ENV_PATH.exists() else ENV_EXAMPLE
    if source.exists():
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()
    return data


def save_env(values: dict[str, str]) -> None:
    # Read existing .env (or .env.example template) to preserve comments and layout
    template = ENV_PATH if ENV_PATH.exists() else ENV_EXAMPLE
    if template.exists():
        lines = template.read_text(encoding="utf-8").splitlines()
        updated_lines: list[str] = []
        handled_keys: set[str] = set()
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k, _ = stripped.split("=", 1)
                k = k.strip()
                if k in values:
                    updated_lines.append(f"{k}={values[k]}")
                    handled_keys.add(k)
                    continue
            updated_lines.append(line)
        # Any remaining new keys appended at end
        for k, v in values.items():
            if k not in handled_keys:
                updated_lines.append(f"{k}={v}")
        ENV_PATH.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    else:
        # Fallback raw write
        content = "\n".join(f"{k}={v}" for k, v in values.items()) + "\n"
        ENV_PATH.write_text(content, encoding="utf-8")


def list_audio_devices() -> tuple[list[str], list[str]]:
    inputs: list[str] = []
    outputs: list[str] = []
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        for d in devices:
            name = d.get("name", "")
            if d.get("max_input_channels", 0) > 0 and name not in inputs:
                inputs.append(name)
            if d.get("max_output_channels", 0) > 0 and name not in outputs:
                outputs.append(name)
    except Exception:
        pass
    return inputs, outputs


def test_llm_connection(base_url: str, api_key: str) -> tuple[bool, str]:
    if not base_url:
        return False, "Base URL kosong"
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"User-Agent": "live-voice-cohost/0.1"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return True, "Koneksi endpoint OK (HTTP 200)"
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return False, "API key salah / ditolak (HTTP 401)"
        return True, f"Endpoint terhubung (HTTP {exc.code})"
    except Exception as exc:
        return False, f"Gagal koneksi: {exc}"


def test_groq_key(api_key: str) -> tuple[bool, str]:
    if not api_key:
        return False, "Key kosong — fallback ke STT lokal / text"
    url = "https://api.groq.com/openai/v1/models"
    req = urllib.request.Request(url, headers={"User-Agent": "live-voice-cohost/0.1"})
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return True, "Groq API key VALID!"
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return False, "Key tidak valid (HTTP 401)"
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, f"Gagal: {exc}"


# ---------------------------------------------------------------------------
# CLI Wizard
# ---------------------------------------------------------------------------
def prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def run_cli() -> None:
    current = load_current_env()
    inputs, outputs = list_audio_devices()

    print("\n" + "=" * 64)
    print("  live-voice-cohost — Setup Wizard (CLI)")
    print("=" * 64)
    print("Tekan Enter untuk mempertahankan nilai default / yang ada.\n")

    # 1. Streamer Name
    cur_name = current.get("JARVIS_STREAMER_NAME", "Pak Febri")
    name = prompt(f"1. Nama Streamer (disapa di live) [{cur_name}]: ") or cur_name

    # 2. LLM Provider
    print("\n2. Pilih Provider LLM:")
    for k, v in PRESETS.items():
        print(f"   [{k}] {v['name']}")
    choice = prompt("Pilihan [1]: ") or "1"
    preset = PRESETS.get(choice, PRESETS["1"])

    cur_base = current.get("JARVIS_LLM_BASE_URL", preset["base_url"])
    base_url = prompt(f"   Base URL [{cur_base}]: ") or cur_base

    cur_model = current.get("JARVIS_LLM_MODEL", preset["model"])
    model = prompt(f"   Model name [{cur_model}]: ") or cur_model

    print(f"   Tip: {preset['key_hint']}")
    cur_llm_key = current.get("JARVIS_LLM_API_KEY", "")
    mask = (cur_llm_key[:4] + "..." + cur_llm_key[-4:]) if len(cur_llm_key) > 8 else (cur_llm_key or "(kosong)")
    llm_key = prompt(f"   LLM API Key [{mask}]: ")
    if not llm_key:
        llm_key = cur_llm_key

    # Ping test
    if base_url:
        print("   Mengecek koneksi LLM...", end=" ", flush=True)
        ok, msg = test_llm_connection(base_url, llm_key)
        print("✓" if ok else "!", msg)

    # 3. STT (Groq fast vs local)
    print("\n3. Speech-to-Text (STT):")
    print("   Groq STT gratis & sangat cepat (~0.9 detik). Rekomendasi pakai Groq.")
    cur_groq = current.get("JARVIS_GROQ_API_KEY", "")
    mask_g = (cur_groq[:4] + "..." + cur_groq[-4:]) if len(cur_groq) > 8 else (cur_groq or "(kosong)")
    groq_key = prompt(f"   Groq API Key (https://console.groq.com/keys) [{mask_g}]: ")
    if not groq_key:
        groq_key = cur_groq

    stt_backend = "groq" if groq_key else current.get("JARVIS_STT_BACKEND", "local")
    if groq_key:
        print("   Mengecek Groq key...", end=" ", flush=True)
        ok, msg = test_groq_key(groq_key)
        print("✓" if ok else "!", msg)

    # 4. Audio In / Out
    print("\n4. Perangkat Audio:")
    cur_in = current.get("JARVIS_INPUT_DEVICE", "")
    print(f"   Mic saat ini: {cur_in or '(default sistem)'}")
    if inputs:
        print("   Perangkat input yang terdeteksi:")
        for idx, dev in enumerate(inputs[:5]):
            print(f"     - {dev}")
    in_dev = prompt(f"   Substring nama Mic [{cur_in}]: ")
    if in_dev == "":
        in_dev = cur_in

    cur_out = current.get("JARVIS_OUTPUT_DEVICE", "")
    print(f"\n   Speaker/OBS saat ini: {cur_out or '(default sistem)'}")
    print("   (Untuk OBS, biasa pakai 'CABLE Input' bila pasang VB-Audio Cable)")
    if outputs:
        print("   Perangkat output yang terdeteksi:")
        for idx, dev in enumerate(outputs[:5]):
            print(f"     - {dev}")
    out_dev = prompt(f"   Substring nama Output [{cur_out}]: ")
    if out_dev == "":
        out_dev = cur_out

    # Save
    updates = {
        "JARVIS_STREAMER_NAME": name,
        "JARVIS_LLM_BASE_URL": base_url,
        "JARVIS_LLM_MODEL": model,
        "JARVIS_LLM_API_KEY": llm_key,
        "JARVIS_STT_BACKEND": stt_backend,
        "JARVIS_GROQ_API_KEY": groq_key,
        "JARVIS_INPUT_DEVICE": in_dev,
        "JARVIS_OUTPUT_DEVICE": out_dev,
    }
    save_env(updates)
    print("\n" + "=" * 64)
    print(f"  ✓ Konfigurasi disimpan ke {ENV_PATH.name}")
    print("  Jalankan `start.bat` untuk menyalakan voice co-host!")
    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# GUI Wizard (Tkinter)
# ---------------------------------------------------------------------------
def run_gui() -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    current = load_current_env()
    inputs, outputs = list_audio_devices()

    root = tk.Tk()
    root.title("Setup — live-voice-cohost")
    root.geometry("640x720")
    root.resizable(False, False)

    # Style
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    padding = {"padx": 16, "pady": 6}

    header = tk.Frame(root, bg="#111827", pady=14)
    header.pack(fill="x")
    title_lbl = tk.Label(header, text="live-voice-cohost — Setup", font=("Segoe UI", 14, "bold"), fg="#F9FAFB", bg="#111827")
    title_lbl.pack()
    sub_lbl = tk.Label(header, text="Konfigurasi cepat co-host live streaming", font=("Segoe UI", 9), fg="#9CA3AF", bg="#111827")
    sub_lbl.pack()

    body = tk.Frame(root, padx=16, pady=10)
    body.pack(fill="both", expand=True)

    # 1. Streamer
    f1 = ttk.LabelFrame(body, text=" 1. Streamer Profile ", padding=8)
    f1.pack(fill="x", **padding)
    ttk.Label(f1, text="Nama streamer (disapa di live):").grid(row=0, column=0, sticky="w")
    streamer_var = tk.StringVar(value=current.get("JARVIS_STREAMER_NAME", "Pak Febri"))
    ttk.Entry(f1, textvariable=streamer_var, width=35).grid(row=0, column=1, padx=6, sticky="ew")

    # 2. LLM
    f2 = ttk.LabelFrame(body, text=" 2. Otak LLM (OpenAI-Compatible) ", padding=8)
    f2.pack(fill="x", **padding)

    ttk.Label(f2, text="Preset Provider:").grid(row=0, column=0, sticky="w")
    preset_var = tk.StringVar(value="1")
    preset_combo = ttk.Combobox(f2, textvariable=preset_var, state="readonly", width=40,
                                values=[f"{k}. {v['name']}" for k, v in PRESETS.items()])
    preset_combo.current(0)
    preset_combo.grid(row=0, column=1, padx=6, pady=2, sticky="ew")

    ttk.Label(f2, text="Base URL:").grid(row=1, column=0, sticky="w")
    base_url_var = tk.StringVar(value=current.get("JARVIS_LLM_BASE_URL", PRESETS["1"]["base_url"]))
    ttk.Entry(f2, textvariable=base_url_var, width=42).grid(row=1, column=1, padx=6, pady=2, sticky="ew")

    ttk.Label(f2, text="Model:").grid(row=2, column=0, sticky="w")
    model_var = tk.StringVar(value=current.get("JARVIS_LLM_MODEL", PRESETS["1"]["model"]))
    ttk.Entry(f2, textvariable=model_var, width=42).grid(row=2, column=1, padx=6, pady=2, sticky="ew")

    ttk.Label(f2, text="API Key:").grid(row=3, column=0, sticky="w")
    llm_key_var = tk.StringVar(value=current.get("JARVIS_LLM_API_KEY", ""))
    ttk.Entry(f2, textvariable=llm_key_var, show="•", width=42).grid(row=3, column=1, padx=6, pady=2, sticky="ew")

    llm_status_lbl = tk.Label(f2, text="Tips: OpenRouter gratis tanpa kartu kredit di openrouter.ai/keys",
                              font=("Segoe UI", 8), fg="#6B7280")
    llm_status_lbl.grid(row=4, column=0, columnspan=2, sticky="w", pady=2)

    def on_preset_change(event=None):
        idx = preset_combo.get().split(".")[0].strip()
        p = PRESETS.get(idx)
        if p and idx != "4":
            base_url_var.set(p["base_url"])
            model_var.set(p["model"])
            llm_status_lbl.config(text=p["key_hint"], fg="#374151")

    preset_combo.bind("<<ComboboxSelected>>", on_preset_change)

    # 3. STT
    f3 = ttk.LabelFrame(body, text=" 3. Speech-to-Text (STT) ", padding=8)
    f3.pack(fill="x", **padding)

    ttk.Label(f3, text="Groq API Key:").grid(row=0, column=0, sticky="w")
    groq_key_var = tk.StringVar(value=current.get("JARVIS_GROQ_API_KEY", ""))
    ttk.Entry(f3, textvariable=groq_key_var, show="•", width=42).grid(row=0, column=1, padx=6, pady=2, sticky="ew")
    stt_tip = tk.Label(f3, text="Groq gratis di console.groq.com/keys (~0.9s). Jika kosong → fallback local whisper.",
                       font=("Segoe UI", 8), fg="#6B7280")
    stt_tip.grid(row=1, column=0, columnspan=2, sticky="w")

    # 4. Audio
    f4 = ttk.LabelFrame(body, text=" 4. Audio I/O ", padding=8)
    f4.pack(fill="x", **padding)

    ttk.Label(f4, text="Microphone:").grid(row=0, column=0, sticky="w")
    in_opts = ["(Default sistem)"] + inputs
    in_var = tk.StringVar(value=current.get("JARVIS_INPUT_DEVICE") or in_opts[0])
    in_combo = ttk.Combobox(f4, textvariable=in_var, values=in_opts, width=38)
    in_combo.grid(row=0, column=1, padx=6, pady=2, sticky="ew")

    ttk.Label(f4, text="Output / OBS:").grid(row=1, column=0, sticky="w")
    out_opts = ["(Default sistem)", "CABLE Input (VB-Audio)"] + outputs
    out_var = tk.StringVar(value=current.get("JARVIS_OUTPUT_DEVICE") or out_opts[0])
    out_combo = ttk.Combobox(f4, textvariable=out_var, values=out_opts, width=38)
    out_combo.grid(row=1, column=1, padx=6, pady=2, sticky="ew")

    # Action buttons
    btn_frame = tk.Frame(root, pady=12)
    btn_frame.pack(fill="x", padx=16)

    def do_test():
        b = base_url_var.get().strip()
        k = llm_key_var.get().strip()
        ok_llm, msg_llm = test_llm_connection(b, k)

        gk = groq_key_var.get().strip()
        ok_groq, msg_groq = test_groq_key(gk) if gk else (True, "STT: Local fallback / text mode")

        status = f"LLM: {msg_llm}\nSTT: {msg_groq}"
        if ok_llm:
            messagebox.showinfo("Hasil Uji Koneksi", status)
        else:
            messagebox.showwarning("Uji Koneksi Gagal", status)

    def do_save():
        in_val = in_var.get().strip()
        if in_val == "(Default sistem)":
            in_val = ""
        out_val = out_var.get().strip()
        if out_val == "(Default sistem)":
            out_val = ""
        gk = groq_key_var.get().strip()
        stt_mode = "groq" if gk else "local"

        updates = {
            "JARVIS_STREAMER_NAME": streamer_var.get().strip() or "Pak Febri",
            "JARVIS_LLM_BASE_URL": base_url_var.get().strip(),
            "JARVIS_LLM_MODEL": model_var.get().strip(),
            "JARVIS_LLM_API_KEY": llm_key_var.get().strip(),
            "JARVIS_STT_BACKEND": stt_mode,
            "JARVIS_GROQ_API_KEY": gk,
            "JARVIS_INPUT_DEVICE": in_val,
            "JARVIS_OUTPUT_DEVICE": out_val,
        }
        save_env(updates)
        messagebox.showinfo("Tersimpan!", f"Konfigurasi berhasil disimpan ke {ENV_PATH.name}.\n\nJalankan start.bat untuk menyalakan!")
        root.destroy()

    test_btn = tk.Button(btn_frame, text="🔍 Tes Koneksi", command=do_test, padx=12, pady=6)
    test_btn.pack(side="left")

    save_btn = tk.Button(btn_frame, text="💾 Simpan Konfigurasi", command=do_save,
                         bg="#2563EB", fg="white", font=("Segoe UI", 10, "bold"), padx=16, pady=6)
    save_btn.pack(side="right")

    root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description="Setup wizard for live-voice-cohost")
    ap.add_argument("--cli", action="store_true", help="Pakai CLI terminal bukan GUI window")
    args = ap.parse_args()

    if args.cli:
        run_cli()
        return

    # Try GUI first, gracefully fallback to CLI
    try:
        run_gui()
    except Exception as exc:
        print(f"[setup] GUI tidak dapat dibuka ({exc}), beralih ke CLI...")
        run_cli()


if __name__ == "__main__":
    main()