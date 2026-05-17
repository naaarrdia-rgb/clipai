import streamlit as st
import os
import json
import re
import subprocess
import tempfile
from pathlib import Path

st.set_page_config(
    page_title="ClipAI - Clips Viraux",
    page_icon="🎬",
    layout="centered"
)

st.markdown("""
<style>
.big-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
.sub { color: #888; margin-top: 0; margin-bottom: 1.5rem; }
.clip-box { border: 1px solid #333; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
.score { background: #1DB954; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🎬 ClipAI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Colle un lien YouTube → reçois des clips TikTok viraux en 9:16 avec sous-titres</p>', unsafe_allow_html=True)

OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

with st.sidebar:
    st.header("⚙️ Paramètres")
    api_key = st.text_input("Clé API Claude (Anthropic)", type="password", placeholder="sk-ant-...")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    st.divider()
    num_clips = st.slider("Nombre de clips", 1, 5, 3)
    clip_duration = st.slider("Durée max (secondes)", 30, 90, 60)
    language = st.selectbox("Langue de la vidéo", ["fr", "en", "ar", "es"], index=0)
    st.divider()
    if st.button("🗑️ Nettoyer les fichiers"):
        import shutil
        shutil.rmtree("outputs", ignore_errors=True)
        shutil.rmtree("temp", ignore_errors=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        TEMP_DIR.mkdir(exist_ok=True)
        st.success("Nettoyé !")

url = st.text_input("🔗 Lien YouTube", placeholder="https://www.youtube.com/watch?v=...")

col1, col2 = st.columns([3,1])
with col1:
    go = st.button("🚀 Générer mes clips", type="primary", use_container_width=True)

if go:
    if not url or not url.startswith("http"):
        st.error("❌ Colle un vrai lien YouTube !")
        st.stop()

    from processor import VideoProcessor
    processor = VideoProcessor(url, num_clips, clip_duration, language)

    with st.status("⏳ Traitement en cours...", expanded=True) as status:

        st.write("📥 Téléchargement de la vidéo YouTube...")
        try:
            video_path, title = processor.download()
            st.write(f"✅ **{title}**")
        except Exception as e:
            st.error(f"❌ Erreur téléchargement : {e}")
            st.stop()

        st.write("🎤 Transcription audio avec Whisper...")
        try:
            transcript = processor.transcribe(video_path)
            st.write(f"✅ {len(transcript)} segments transcrits")
        except Exception as e:
            st.error(f"❌ Erreur transcription : {e}")
            st.stop()

        st.write("🤖 Détection des moments viraux par l'IA...")
        try:
            clips_info = processor.detect_viral_moments(transcript)
            st.write(f"✅ {len(clips_info)} moments sélectionnés")
        except Exception as e:
            st.error(f"❌ Erreur IA : {e}")
            st.stop()

        st.write("✂️ Création des clips en format 9:16 + sous-titres...")
        try:
            output_files = processor.create_clips(video_path, clips_info, transcript)
            st.write(f"✅ {len(output_files)} clips prêts !")
        except Exception as e:
            st.error(f"❌ Erreur montage : {e}")
            st.stop()

        status.update(label="✅ Clips prêts à télécharger !", state="complete")

    st.success(f"🎉 {len(output_files)} clips générés !")
    st.divider()
    st.subheader("📥 Télécharge tes clips")

    for i, (clip_path, info) in enumerate(zip(output_files, clips_info)):
        with st.container():
            c1, c2 = st.columns([4,1])
            with c1:
                st.markdown(f"**Clip {i+1}** — {info.get('title','Moment viral')}")
                st.caption(f"⏱️ {info.get('start',0):.0f}s → {info.get('end',0):.0f}s | {info.get('reason','')}")
                st.caption(f"📝 {info.get('description','')}")
                st.caption(f"#️⃣ {info.get('hashtags','')}")
            with c2:
                score = info.get('viral_score', 8)
                st.markdown(f'<span class="score">🔥 {score}/10</span>', unsafe_allow_html=True)

            if os.path.exists(clip_path):
                st.video(clip_path)
                with open(clip_path, "rb") as f:
                    st.download_button(
                        f"⬇️ Télécharger clip {i+1}",
                        f,
                        file_name=f"clip_{i+1}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            st.divider()
