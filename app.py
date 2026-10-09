import json
import os
import re
import threading
import uuid

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, render_template


# =========================================================
# VORTX AI TEXT - Railway Flask Server
# =========================================================

PROVIDER_URL = "https://ptero.pro"
CHAT_URL = f"{PROVIDER_URL}/wp-json/mlp/v1/chat-stream"

DEFAULT_MODEL = "deepseek-v4.1-flash:free"

MODELS = {
    "ling-3.0-flash": {
        "id": "ling-3.0-flash",
        "name": "Ling 3.0 Flash",
    },
    "deepseek-v4.1-flash:free": {
        "id": "deepseek-v4.1-flash:free",
        "name": "DeepSeek v4.1 Flash",
    },
    "atria-dawn-preview": {
        "id": "atria-dawn-preview",
        "name": "Atria-Dawn-Preview",
    },
    "agnes-2.5-flash": {
        "id": "agnes-2.5-flash",
        "name": "Agnes 2.5 Flash",
    },
    "agnes-3.0-flash": {
        "id": "agnes-3.0-flash",
        "name": "Agnes 3.0 Flash",
    },
    "laguna-s-2.1": {
        "id": "laguna-s-2.1",
        "name": "Laguna S 2.1",
    },
}

MODEL_ALIASES = {
    "1": "ling-3.0-flash",
    "2": "deepseek-v4.1-flash:free",
    "3": "atria-dawn-preview",
    "4": "agnes-2.5-flash",
    "5": "agnes-3.0-flash",
    "6": "laguna-s-2.1",
    "ling": "ling-3.0-flash",
    "deepseek": "deepseek-v4.1-flash:free",
    "atria": "atria-dawn-preview",
    "agnes-2.5": "agnes-2.5-flash",
    "agnes-3.0": "agnes-3.0-flash",
    "laguna": "laguna-s-2.1",
}

USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 10; K) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/120.0.0.0 "
    "Mobile Safari/537.36"
)

app = Flask(__name__)

session_lock = threading.Lock()
session_data = None


# =========================================================
# MODEL
# =========================================================

def resolve_model(value):
    if value is None:
        return DEFAULT_MODEL

    value = str(value).strip()

    if not value:
        return DEFAULT_MODEL

    if value in MODELS:
        return value

    lowered = value.lower()

    if lowered in MODEL_ALIASES:
        return MODEL_ALIASES[lowered]

    for model_id, info in MODELS.items():
        if lowered == info["name"].lower():
            return model_id

    return DEFAULT_MODEL


# =========================================================
# PROVIDER SESSION
# =========================================================

def fetch_real_session():
    session = requests.Session()

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": (
            "ar-EG,ar;q=0.9,en-US;q=0.8,en;q=0.7"
        ),
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    response = session.get(
        PROVIDER_URL,
        headers=headers,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to open provider page: HTTP {response.status_code}"
        )

    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    nonce = None

    patterns = [
        r'["\']nonce["\']\s*:\s*["\']([a-f0-9]+)["\']',
        r'mlp_nonce["\']?\s*[:=]\s*["\']([a-f0-9]+)["\']',
        r'wp_nonce["\']?\s*[:=]\s*["\']([a-f0-9]+)["\']',
        r'x-wp-nonce["\']?\s*[:=]\s*["\']([a-f0-9]+)["\']',
        r'data-nonce=["\']([a-f0-9]+)["\']',
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            html,
            re.IGNORECASE,
        )

        if match:
            nonce = match.group(1)
            break

    if not nonce:
        for script in soup.find_all("script"):
            script_text = (
                script.string or
                script.get_text()
            )

            if "nonce" not in script_text.lower():
                continue

            match = re.search(
                r'["\']([a-f0-9]{10,})["\']',
                script_text,
                re.IGNORECASE,
            )

            if match:
                nonce = match.group(1)
                break

    if not nonce:
        raise RuntimeError(
            "Provider nonce was not found."
        )

    guest_token = None

    for cookie in session.cookies:
        if any(
            key in cookie.name.lower()
            for key in ("mlp", "guest", "token")
        ):
            guest_token = cookie.value
            break

    if not guest_token:
        match = re.search(
            r'guest[_-]?token["\']?\s*[:=]\s*["\']([a-f0-9]+)["\']',
            html,
            re.IGNORECASE,
        )

        guest_token = (
            match.group(1)
            if match
            else uuid.uuid4().hex
        )

    return {
        "session": session,
        "nonce": nonce,
        "guest_token": guest_token,
        "conversation_id": str(uuid.uuid4()),
    }


def get_session():
    global session_data

    with session_lock:
        if session_data is None:
            session_data = fetch_real_session()

        return session_data


def reset_session():
    global session_data

    with session_lock:
        session_data = None


# =========================================================
# CHAT
# =========================================================

def chat_with_model(
    model_id,
    user_message,
    current_session,
    attachments=None,
):
    payload = {
        "message": user_message,
        "model": model_id,
        "conversation_id": current_session["conversation_id"],
        "attachments": attachments or [],
        "history": [],
        "lang": "ar",
        "github_repo": "",
        "mode": "quick",
    }

    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Content-Type": "application/json",
        "sec-ch-ua-platform": '"Android"',
        "sec-ch-ua": '"Chromium";v="120", "Google Chrome";v="120"',
        "sec-ch-ua-mobile": "?1",
        "x-wp-nonce": current_session["nonce"],
        "x-mlp-guest-token": current_session["guest_token"],
        "x-mlp-guest-username": (
            f"user_{current_session['conversation_id'][:6]}"
        ),
        "origin": PROVIDER_URL,
        "sec-fetch-site": "same-origin",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
        "referer": f"{PROVIDER_URL}/",
        "accept-language": (
            "ar-EG,ar;q=0.9,en-US;q=0.8,en;q=0.7"
        ),
    }

    response = current_session["session"].post(
        CHAT_URL,
        json=payload,
        headers=headers,
        stream=True,
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Provider returned HTTP {response.status_code}"
        )

    full_reply = ""
    meta = None

    for line in response.iter_lines():

        if not line:
            continue

        if isinstance(line, bytes):
            line = line.decode(
                "utf-8",
                errors="replace",
            )

        if not line.startswith("data:"):
            continue

        data_str = line[5:].strip()

        if data_str == "keepalive":
            continue

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if "token" in data:
            full_reply += str(data["token"])

        if data.get("done"):
            meta = data

            if data.get("conversation_id"):
                current_session["conversation_id"] = (
                    data["conversation_id"]
                )

            break

    if not full_reply.strip():
        raise RuntimeError(
            "Provider returned an empty response."
        )

    return full_reply, meta


def ask_provider(
    model_id,
    user_message,
    attachments=None,
):
    current = get_session()

    try:
        return chat_with_model(
            model_id,
            user_message,
            current,
            attachments,
        )

    except Exception:
        reset_session()

    current = get_session()

    return chat_with_model(
        model_id,
        user_message,
        current,
        attachments,
    )


# =========================================================
# WEBSITE
# =========================================================

@app.get("/")
def home():
    return render_template("index.html")


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "status": "running",
            "server": "VORTX AI Flask",
        }
    )


# =========================================================
# MODELS
# =========================================================

@app.get("/api/models")
def models():
    return jsonify(
        {
            "ok": True,
            "models": list(MODELS.values()),
            "default_model": DEFAULT_MODEL,
        }
    )


# =========================================================
# DIRECT API
# =========================================================

@app.post("/api/convert")
def convert():

    try:
        body = request.get_json(
            silent=True
        ) or {}

        text = body.get("text")

        if text is None:
            text = body.get("message")

        if text is None:
            return jsonify(
                {
                    "ok": False,
                    "error": "Missing 'text' field.",
                }
            ), 400

        user_message = str(text).strip()

        if not user_message:
            return jsonify(
                {
                    "ok": False,
                    "error": "Text is empty.",
                }
            ), 400

        model_id = resolve_model(
            body.get(
                "model",
                body.get("selected_model"),
            )
        )

        incoming_attachments = body.get(
            "attachments",
            [],
        )

        if not isinstance(
            incoming_attachments,
            list,
        ):
            incoming_attachments = []

        attachments = []

        for item in incoming_attachments[:5]:

            if not isinstance(item, dict):
                continue

            name = str(
                item.get(
                    "name",
                    "file",
                )
            )

            mime = str(
                item.get(
                    "mime",
                    "application/octet-stream",
                )
            )

            data = item.get(
                "data_base64",
                "",
            )

            if not isinstance(data, str):
                continue

            if not data:
                continue

            if len(data) > 9_000_000:
                continue

            attachments.append(
                {
                    "name": name,
                    "mime": mime,
                    "data_base64": data,
                }
            )

        answer, meta = ask_provider(
            model_id,
            user_message,
            attachments,
        )

        return jsonify(
            {
                "ok": True,
                "answer": answer,
                "model": model_id,
                "model_name": MODELS[
                    model_id
                ]["name"],
                "conversation_id": (
                    meta.get(
                        "conversation_id"
                    )
                    if isinstance(
                        meta,
                        dict,
                    )
                    else None
                ),
            }
        )

    except requests.exceptions.Timeout:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Provider request timed out."
                ),
            }
        ), 504

    except requests.exceptions.RequestException as exc:

        return jsonify(
            {
                "ok": False,
                "error": (
                    f"Provider request failed: {exc}"
                ),
            }
        ), 502

    except Exception as exc:

        return jsonify(
            {
                "ok": False,
                "error": str(exc),
            }
        ), 500


# =========================================================
# WEBSITE PROCESS API
# =========================================================

@app.post("/api/process")
def process():

    data = request.get_json(
        silent=True
    ) or {}

    text = str(
        data.get(
            "text",
            "",
        )
    ).strip()

    action = str(
        data.get(
            "action",
            "improve",
        )
    ).strip()

    if not text:
        return jsonify(
            {
                "ok": False,
                "error": "اكتب النص أولاً.",
            }
        ), 400

    prompts = {

        "improve":
            "حسّن النص التالي مع الحفاظ على المعنى، "
            "واجعله واضحًا وطبيعيًا واحترافيًا:",

        "summarize":
            "لخّص النص التالي باختصار مع الحفاظ "
            "على أهم النقاط:",

        "fusha":
            "حوّل النص التالي من العامية العربية إلى "
            "العربية الفصحى السليمة مع الحفاظ على المعنى:",

        "rewrite":
            "أعد صياغة النص التالي بأسلوب مختلف "
            "وواضح واحترافي مع الحفاظ على المعنى:",

        "post":
            "حوّل الفكرة التالية إلى منشور عربي جذاب "
            "ومرتب مناسب لقناة Telegram، بدون مبالغة "
            "أو معلومات غير موجودة:",
    }

    prompt = (
        prompts.get(
            action,
            prompts["improve"],
        )
        + "\n\n"
        + text
    )

    try:

        answer, meta = ask_provider(
            DEFAULT_MODEL,
            prompt,
        )

        return jsonify(
            {
                "ok": True,
                "answer": answer,
                "model": MODELS[
                    DEFAULT_MODEL
                ]["name"],
                "conversation_id": (
                    meta.get(
                        "conversation_id"
                    )
                    if isinstance(
                        meta,
                        dict,
                    )
                    else None
                ),
            }
        )

    except requests.exceptions.Timeout:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "انتهت مهلة الطلب. حاول مرة أخرى."
                ),
            }
        ), 504

    except requests.exceptions.RequestException as exc:

        return jsonify(
            {
                "ok": False,
                "error": (
                    f"فشل الاتصال بخادم AI: {exc}"
                ),
            }
        ), 502

    except Exception as exc:

        return jsonify(
            {
                "ok": False,
                "error": str(exc),
            }
        ), 500


# =========================================================
# LOCAL TEST
# Railway uses Gunicorn through Procfile.
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8080,
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )