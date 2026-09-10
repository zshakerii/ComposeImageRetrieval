import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Image Retrieval Studio", page_icon="🔎", layout="wide")

BACKEND_DEFAULT = "./corrected_image_retrieval_align_base.py"

MODEL_LABELS = {
    "align": "ALIGN — align-base",
    "clip": "CLIP — ViT-B/32",
    "open_clip": "OpenCLIP — ViT-H-14",
    "siglip": "SigLIP / SigLIP2",
    "blip": "BLIP-ITM",
    "searle": "SEARLE",
    "clip_beta": "CLIP-β",
    "qwen3": "Qwen3-VL-Embedding",
}
INTERACTIVE_SUPPORTED = {"align", "clip", "open_clip", "siglip"}


def load_backend(path: str):
    path = str(Path(path).expanduser().resolve())
    if not Path(path).is_file():
        raise FileNotFoundError(f"Backend file not found: {path}")
    spec = importlib.util.spec_from_file_location("retrieval_backend", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load backend: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner=False)
def cached_backend(path: str):
    return load_backend(path)


@st.cache_resource(show_spinner=False)
def cached_model(backend_path: str, kind: str, model_path: str, clip_model: str, open_clip_model: str, open_clip_pretrained: str):
    b = cached_backend(backend_path)
    if kind == "align":
        return b.load_align_model(model_path)
    if kind == "clip":
        return b.load_clip_model(clip_model)
    if kind == "open_clip":
        model, preprocess = b.load_open_clip_model(open_clip_model, open_clip_pretrained)
        # tokenizer is needed for text embedding
        import open_clip
        tokenizer = open_clip.get_tokenizer(open_clip_model)
        return model, preprocess, tokenizer
    if kind == "siglip":
        return b.load_siglip_model(model_path)
    raise ValueError(f"Interactive mode is not implemented for {kind}")


def list_gallery_ids(folder: str):
    b = cached_backend(st.session_state.backend_path)
    return b.scan_gallery_ids(folder)


def get_image_path(folder: str, image_id: str):
    b = cached_backend(st.session_state.backend_path)
    return b.find_image_path(folder, image_id)


def make_gallery_cache(kind, folder, cache_dir, batch_size, model_path, clip_model, open_clip_model, open_clip_pretrained):
    b = cached_backend(st.session_state.backend_path)
    gallery_ids = list_gallery_ids(folder)
    cache_key = f"ui-{kind}-{Path(folder).resolve()}"
    dataset = "interactive"
    split = "test"

    if kind == "align":
        model, processor = cached_model(st.session_state.backend_path, kind, model_path, clip_model, open_clip_model, open_clip_pretrained)
        features = b.build_align_image_cache(
            gallery_ids, folder, model, processor, cache_dir, dataset, split,
            model_key="align-base", batch_size=batch_size, force_rebuild=False
        )
    elif kind == "clip":
        model, preprocess = cached_model(st.session_state.backend_path, kind, model_path, clip_model, open_clip_model, open_clip_pretrained)
        cache_path = b.clip_embedding_cache_path(cache_dir, clip_model, dataset, split)
        partial = cache_path.with_suffix(cache_path.suffix + ".partial")
        norm, raw = b.build_clip_image_cache(
            gallery_ids, folder, model, preprocess, partial,
            model_name=clip_model, checkpoint_every=250, cache_dataset=dataset, cache_split=split
        )
        features = norm
    elif kind == "open_clip":
        model, preprocess, _tokenizer = cached_model(st.session_state.backend_path, kind, model_path, clip_model, open_clip_model, open_clip_pretrained)
        features = b.build_open_clip_image_cache(
            gallery_ids, folder, model, preprocess, cache_dir, dataset, split,
            model_key=f"open_clip-{open_clip_model}-{open_clip_pretrained}", batch_size=batch_size, force_rebuild=False
        )
    elif kind == "siglip":
        model, processor = cached_model(st.session_state.backend_path, kind, model_path, clip_model, open_clip_model, open_clip_pretrained)
        features = b.build_siglip_image_cache(
            gallery_ids, folder, model, processor, cache_dir, dataset, split,
            model_key=f"siglip-{Path(model_path).name}", batch_size=batch_size, force_rebuild=False
        )
    else:
        raise ValueError(f"Unsupported interactive model: {kind}")

    ids = list(gallery_ids)
    matrix = np.vstack([features[x].detach().cpu().numpy().reshape(-1) for x in ids]).astype("float32")
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return ids, matrix


def encode_uploaded_image_and_text(kind, image, text, model_path, clip_model, open_clip_model, open_clip_pretrained):
    b = cached_backend(st.session_state.backend_path)
    model_obj = cached_model(st.session_state.backend_path, kind, model_path, clip_model, open_clip_model, open_clip_pretrained)
    if kind == "align":
        model, processor = model_obj
        return b.align_image_embedding(image, model, processor), b.align_text_embedding(text, model, processor)
    if kind == "clip":
        model, preprocess = model_obj
        return b.clip_image_embedding(image, model, preprocess), b.clip_text_embedding(text, model)
    if kind == "open_clip":
        model, preprocess, tokenizer = model_obj
        return b.open_clip_image_embedding(image, model, preprocess), b.open_clip_text_embedding(text, model, tokenizer)
    if kind == "siglip":
        model, processor = model_obj
        return b.siglip_image_embedding(image, model, processor), b.siglip_text_embedding(text, model, processor)
    raise ValueError(f"Unsupported interactive model: {kind}")


def run_evaluation(args):
    cmd = [sys.executable, st.session_state.backend_path]
    for k, v in args.items():
        if v is None or v == "" or v is False:
            continue
        flag = "--" + k
        if v is True:
            cmd.append(flag)
        elif isinstance(v, list):
            cmd.append(flag)
            cmd.extend(map(str, v))
        else:
            cmd.extend([flag, str(v)])

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(st.session_state.backend_path).resolve().parent))
    return proc.returncode, proc.stdout, proc.stderr


def inject_css():
    st.markdown("""
    <style>
    .main-title {font-size: 2.2rem; font-weight: 800; margin-bottom: .2rem;}
    .subtitle {color:#6b7280; margin-bottom:1.4rem;}
    .section-card {padding:1rem 1.15rem; border:1px solid rgba(128,128,128,.22); border-radius:14px; background:rgba(128,128,128,.045);}
    .result-card {padding:.55rem; border:1px solid rgba(128,128,128,.2); border-radius:12px; height:100%;}
    .rank {font-weight:800; font-size:1.05rem;}
    </style>
    """, unsafe_allow_html=True)


inject_css()
if "backend_path" not in st.session_state:
    st.session_state.backend_path = BACKEND_DEFAULT

st.markdown('<div class="main-title">🔎 Image Retrieval Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">رابط تعاملی برای ارزیابی مدل‌های Image Retrieval و تست مستقیم با عکس و متن</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ تنظیمات پروژه")
    st.session_state.backend_path = st.text_input("مسیر فایل Backend", st.session_state.backend_path)
    st.caption("فایل backend همان corrected_image_retrieval_align_base.py است.")

mode = st.radio("۱) نوع عملیات را انتخاب کنید", ["ارزیابی (Evaluation)", "تست تعاملی (Interactive Test)"], horizontal=True)

st.divider()
model_kind = st.selectbox("۲) نوع مدل", list(MODEL_LABELS.keys()), format_func=lambda x: MODEL_LABELS[x])

if mode.startswith("ارزیابی"):
    st.markdown("### 📊 ارزیابی روی Dataset")
    c1, c2 = st.columns(2)
    with c1:
        dataset = st.selectbox("Dataset", ["cirr", "circo"])
        image_folder = st.text_input("Image Folder", "./cirr-val")
        json_path = st.text_input("JSON Path", "./cirr-val.json")
        split = st.selectbox("Split", ["auto", "train", "val", "test", "unknown"], index=2)
        cache_dir = st.text_input("Embedding Cache", "./embedding_cache")
    with c2:
        subset = st.checkbox("CIRR Subset Protocol", value=(dataset == "cirr"))
        require_gt = st.checkbox("Require CIRR Ground Truth", value=False)
        strict = st.checkbox("Strict Evaluation", value=False)
        alphas = st.multiselect("Alpha", [0.0, .25, .5, .75, 1.0], default=[0.0, .25, .5, .75, 1.0])

    model_path = None
    if model_kind in {"align", "siglip", "blip", "qwen3"}:
        defaults = {"align": "./models_download/align-base", "siglip": "", "blip": "", "qwen3": "./models_download/Qwen3-VL-Embedding"}
        model_path = st.text_input(f"{model_kind} model path", defaults[model_kind])

    if model_kind == "clip":
        st.text_input("CLIP model", "ViT-B/32", key="eval_clip_model")
    if model_kind == "open_clip":
        st.text_input("OpenCLIP model", "ViT-H-14", key="eval_open_clip_model")
        st.text_input("OpenCLIP pretrained", "laion2b_s32b_b79k", key="eval_open_clip_pretrained")

    b1, b2 = st.columns([1, 4])
    with b1:
        start = st.button("🚀 شروع ارزیابی", type="primary", use_container_width=True)
    if start:
        args = {
            "dataset": dataset,
            "image_folder": image_folder,
            "json_path": json_path,
            "split": split,
            "models": [model_kind],
            "alphas": alphas or [0.5],
            "embedding_cache_dir": cache_dir,
            "cirr_subset": subset,
            "require_cirr_gt": require_gt,
            "strict_evaluation": strict,
        }
        if model_path:
            flag_map = {"align": "align_path", "siglip": "siglip_path", "blip": "blip_path", "qwen3": "qwen3_path"}
            if model_kind in flag_map:
                args[flag_map[model_kind]] = model_path
        if model_kind == "clip":
            args["clip_model"] = st.session_state.get("eval_clip_model", "ViT-B/32")
        if model_kind == "open_clip":
            args["open_clip_model"] = st.session_state.get("eval_open_clip_model", "ViT-H-14")
            args["open_clip_pretrained"] = st.session_state.get("eval_open_clip_pretrained", "laion2b_s32b_b79k")
        with st.spinner("در حال اجرای ارزیابی..."):
            code, out, err = run_evaluation(args)
        st.session_state.eval_output = out
        st.session_state.eval_error = err
        st.session_state.eval_code = code

    if "eval_output" in st.session_state:
        st.subheader("خروجی ارزیابی")
        if st.session_state.eval_code == 0:
            st.success("ارزیابی با موفقیت پایان یافت")
        else:
            st.error(f"فرآیند با کد {st.session_state.eval_code} پایان یافت")
        st.code(st.session_state.eval_output or "(no stdout)", language="text")
        if st.session_state.eval_error:
            with st.expander("خطا / STDERR"):
                st.code(st.session_state.eval_error, language="text")

else:
    st.markdown("### 🧪 تست تعاملی")
    if model_kind not in INTERACTIVE_SUPPORTED:
        st.warning(f"تست تعاملی در این UI برای {MODEL_LABELS[model_kind]} فعال نشده است. برای این مدل از بخش ارزیابی استفاده کنید.")
    else:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### ورودی Query")
            uploaded = st.file_uploader("عکس مرجع را بارگذاری کنید", type=["jpg", "jpeg", "png", "webp", "bmp"])
            text = st.text_area("متن اصلاحی / Relative Caption", placeholder="مثلاً: a black shirt instead of a red one")
            alpha = st.slider("وزن متن (α)", 0.0, 1.0, 0.5, 0.05)
            top_k = st.slider("Top-K", 1, 50, 10)
            gallery_folder = st.text_input("Gallery Folder", "./cirr-val")
            cache_dir = st.text_input("Embedding Cache", "./embedding_cache", key="test_cache")
            batch_size = st.number_input("Gallery batch size", min_value=1, max_value=128, value=8)

            if model_kind in {"align", "siglip"}:
                model_path = st.text_input("Model Path", "./models_download/align-base" if model_kind == "align" else "")
            else:
                model_path = ""
            clip_model = st.text_input("CLIP model", "ViT-B/32") if model_kind == "clip" else "ViT-B/32"
            open_clip_model = st.text_input("OpenCLIP model", "ViT-H-14") if model_kind == "open_clip" else "ViT-H-14"
            open_clip_pretrained = st.text_input("OpenCLIP pretrained", "laion2b_s32b_b79k") if model_kind == "open_clip" else "laion2b_s32b_b79k"
            run = st.button("🔍 جستجو در گالری", type="primary", use_container_width=True)

        with right:
            st.markdown("#### Query Preview")
            if uploaded:
                qimg = Image.open(uploaded).convert("RGB")
                st.image(qimg, caption="Reference / Query Image", use_container_width=True)
            else:
                st.info("یک تصویر برای تست بارگذاری کنید.")
            if text.strip():
                st.markdown("**Text:**")
                st.write(text)

        if run:
            if uploaded is None or not text.strip():
                st.error("لطفاً هم تصویر و هم متن را وارد کنید.")
            elif not Path(gallery_folder).is_dir():
                st.error(f"Gallery folder پیدا نشد: {gallery_folder}")
            else:
                try:
                    qimg = Image.open(uploaded).convert("RGB")
                    with st.status("در حال آماده‌سازی مدل و embedding گالری...", expanded=True) as status:
                        image_feat, text_feat = encode_uploaded_image_and_text(
                            model_kind, qimg, text, model_path, clip_model, open_clip_model, open_clip_pretrained
                        )
                        ids, gallery = make_gallery_cache(
                            model_kind, gallery_folder, cache_dir, int(batch_size), model_path,
                            clip_model, open_clip_model, open_clip_pretrained
                        )
                        alpha = float(alpha)
                        query = alpha * image_feat + (1.0 - alpha) * text_feat
                        query = query.float().numpy().reshape(-1)
                        query /= max(float(np.linalg.norm(query)), 1e-12)
                        sims = gallery @ query
                        order = np.argsort(-sims)[:top_k]
                        results = [(ids[i], float(sims[i])) for i in order]
                        status.update(label="جستجو با موفقیت انجام شد", state="complete")
                    st.session_state.test_results = results
                    st.session_state.test_gallery_folder = gallery_folder
                except Exception as exc:
                    st.exception(exc)

        if "test_results" in st.session_state:
            st.divider()
            st.subheader(f"Top-{len(st.session_state.test_results)} Results")
            results = st.session_state.test_results
            cols = st.columns(min(5, len(results)) or 1)
            for idx, (image_id, score) in enumerate(results):
                col = cols[idx % len(cols)]
                path = get_image_path(st.session_state.test_gallery_folder, image_id)
                with col:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    if path and Path(path).is_file():
                        st.image(path, use_container_width=True)
                    st.markdown(f'<div class="rank">#{idx+1}</div>', unsafe_allow_html=True)
                    st.caption(f"ID: {image_id}\nSimilarity: {score:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("Image Retrieval Studio · ساخته شده برای pipeline ارزیابی CIRR/CIRCO و تست تعاملی مدل‌های retrieval")
