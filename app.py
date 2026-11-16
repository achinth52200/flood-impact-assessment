"""
Streamlit UI — Flood Impact Assessment
Launch: streamlit run app.py
"""
import io, sys, tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import tifffile
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="Flood Impact Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0d1117}
.stMetric{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}
</style>""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model(ckpt_path):
    from models.unet import build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model().to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device, float(ckpt.get("best_iou", 0))


def read_tif_upload(uploaded) -> np.ndarray:
    """Uploaded file → (C,H,W) float32."""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(uploaded.read()); pth = tmp.name
    arr = tifffile.imread(pth).astype(np.float32)
    Path(pth).unlink(missing_ok=True)
    if arr.ndim == 2: arr = arr[np.newaxis]
    if arr.shape[-1] < arr.shape[0]: pass          # already (C,H,W)
    elif arr.ndim == 3 and arr.shape[2] < arr.shape[0]:
        arr = np.transpose(arr,(2,0,1))
    return arr


def norm(x):
    lo,hi = np.percentile(x,[2,98])
    return np.clip((x-lo)/max(hi-lo,1e-6),0,1)


def to_rgb(arr):
    # channels: 0=S1vv,1=S1vh,2=B2,3=B3,4=B4,5=B8,6=B11
    # natural colour: R=B4(ch4), G=B3(ch3), B=B2(ch2)
    C = arr.shape[0]
    r,g,b = (4,3,2) if C>4 else (0,0,0)
    rgb = np.stack([norm(arr[r]),norm(arr[g]),norm(arr[b])],axis=-1)
    return (rgb*255).astype(np.uint8)


@torch.no_grad()
def predict(model, device, arr, threshold=0.5):
    from config import NUM_CHANNELS
    a = arr[:NUM_CHANNELS].copy()
    for c in range(a.shape[0]):
        lo,hi = a[c].min(),a[c].max()
        if hi > 1.0: a[c] = np.clip((a[c]-lo)/max(hi-lo,1e-6),0,1)
    t    = torch.from_numpy(a).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
    return (prob>=threshold).astype(np.uint8), prob


def overlay_fig(rgb, mask, gt=None, title=""):
    n = 3 if gt is not None else 2
    fig,axes = plt.subplots(1,n,figsize=(5*n,5),facecolor="#0d1117")
    fig.suptitle(title,color="white",fontsize=12,fontweight="bold")
    for ax in axes: ax.axis("off"); ax.set_facecolor("#0d1117")

    axes[0].imshow(rgb)
    axes[0].set_title("Satellite RGB",color="#e6edf3",fontsize=11)

    axes[1].imshow(rgb)
    ov = np.zeros((*mask.shape,4)); ov[mask==1]=(0.1,0.5,1.0,0.65)
    axes[1].imshow(ov,interpolation="nearest")
    axes[1].set_title(f"Predicted Flood ({mask.mean()*100:.1f}%)",
                      color="#e6edf3",fontsize=11)

    if gt is not None:
        tp=(mask==1)&(gt==1); fp=(mask==1)&(gt==0); fn=(mask==0)&(gt==1)
        cf=np.zeros((*mask.shape,4))
        cf[tp]=(0.1,0.5,1.0,0.75); cf[fp]=(1.0,0.1,0.1,0.75); cf[fn]=(1.0,0.6,0.0,0.75)
        axes[2].imshow(rgb); axes[2].imshow(cf,interpolation="nearest")
        axes[2].set_title("TP / FP / FN",color="#e6edf3",fontsize=11)
        axes[2].legend(handles=[
            mpatches.Patch(color=(0.1,0.5,1.0),label="True Positive"),
            mpatches.Patch(color=(1.0,0.1,0.1),label="False Positive"),
            mpatches.Patch(color=(1.0,0.6,0.0),label="False Negative"),
        ],loc="lower right",fontsize=8,facecolor="#161b22",
          labelcolor="white",edgecolor="#30363d")

    plt.tight_layout()
    return fig


def metrics(pred,gt,eps=1e-6):
    p,t = pred.flatten().astype(bool), gt.flatten().astype(bool)
    tp,fp,fn,tn = (p&t).sum(),(p&~t).sum(),(~p&t).sum(),(~p&~t).sum()
    return dict(
        IoU  =float((tp+eps)/(tp+fp+fn+eps)),
        F1   =float((2*tp+eps)/(2*tp+fp+fn+eps)),
        Prec =float((tp+eps)/(tp+fp+eps)),
        Recall=float((tp+eps)/(tp+fn+eps)),
        Acc  =float((tp+tn)/(tp+fp+fn+tn+eps)),
    )


def fig_png(fig):
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=150,
        bbox_inches="tight",facecolor=fig.get_facecolor()); buf.seek(0)
    return buf


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌊 Flood Assessment")
    st.divider()
    page = st.radio("", ["🔍 Predict","📊 Evaluate","📈 Logs","ℹ️ About"],
                    label_visibility="collapsed")
    st.divider()
    from config import BEST_CKPT, MODEL_DIR
    ckpts = sorted(MODEL_DIR.glob("*.pth"))
    if ckpts:
        sel_ckpt = st.selectbox("Checkpoint",
                                [str(p) for p in ckpts],
                                format_func=lambda x: Path(x).name)
    else:
        sel_ckpt = None
        st.warning("No checkpoint found.\n`python train.py`")
    st.divider()
    threshold = st.slider("Flood threshold", 0.1, 0.9, 0.5, 0.05)


# ── Predict ───────────────────────────────────────────────────────────────────

if "Predict" in page:
    st.title("🔍 Flood Prediction")

    tab1,tab2 = st.tabs(["Processed Tile (7-band)","Raw S1 + S2"])

    with tab1:
        c1,c2 = st.columns(2)
        with c1: up = st.file_uploader("7-channel tile (.tif)",type=["tif","tiff"])
        with c2: gt_up = st.file_uploader("Ground truth mask (optional)",type=["tif","tiff"])

        if up and sel_ckpt:
            model,device,best_iou = load_model(sel_ckpt)
            arr = read_tif_upload(up)
            st.caption(f"Shape: {arr.shape}  |  Device: {device}  |  Model val-IoU: {best_iou:.4f}")

            with st.spinner("Running inference…"):
                mask,prob = predict(model,device,arr,threshold)

            gt = None
            if gt_up:
                g = read_tif_upload(gt_up)
                gt = (g[0] if g.ndim==3 else g).astype(np.uint8)

            # Metric cards
            m_vals = metrics(mask,gt) if gt is not None else {}
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Flooded Area",f"{mask.mean()*100:.1f}%")
            c2.metric("Max Prob",   f"{prob.max():.3f}")
            if m_vals:
                c3.metric("IoU",  f"{m_vals['IoU']:.4f}")
                c4.metric("F1",   f"{m_vals['F1']:.4f}")

            # Overlay
            rgb = to_rgb(arr)
            fig = overlay_fig(rgb,mask,gt,title=up.name)
            st.pyplot(fig,use_container_width=True); plt.close(fig)

            # Prob heatmap
            with st.expander("Probability heatmap"):
                f2,ax2=plt.subplots(figsize=(5,4),facecolor="#0d1117")
                im=ax2.imshow(prob,cmap="Blues",vmin=0,vmax=1)
                ax2.set_title("Flood probability",color="white"); ax2.axis("off")
                plt.colorbar(im,ax=ax2,fraction=0.046,pad=0.04)
                f2.patch.set_facecolor("#0d1117")
                st.pyplot(f2,use_container_width=True); plt.close(f2)

            st.download_button("⬇ Download PNG", fig_png(fig),
                               Path(up.name).stem+"_flood.png","image/png")
        elif up and not sel_ckpt:
            st.warning("Train first: `python train.py`")

    with tab2:
        st.info("Upload a raw Sentinel-1 (2-band) and Sentinel-2 (13-band) pair. "
                "Preprocessing (SAR dB scaling, band selection, resampling) runs automatically.")
        s1_up = st.file_uploader("Sentinel-1 (.tif)",type=["tif","tiff"],key="s1")
        s2_up = st.file_uploader("Sentinel-2 (.tif)",type=["tif","tiff"],key="s2")
        if s1_up and s2_up and sel_ckpt:
            from config import S2_BAND_IDX
            import cv2 as _cv2
            model,device,_ = load_model(sel_ckpt)
            s1 = read_tif_upload(s1_up)
            s2 = read_tif_upload(s2_up)
            # dB scale S1
            s1 = np.where(s1>0, 10*np.log10(np.clip(s1,1e-10,None)), -30.)
            s1 = np.clip((s1+30)/30, 0, 1).astype(np.float32)
            # select S2 bands
            if s2.shape[0] >= max(S2_BAND_IDX)+1:
                s2 = s2[S2_BAND_IDX]
            for c in range(s2.shape[0]):
                lo,hi=np.percentile(s2[c],[2,98]); s2[c]=np.clip((s2[c]-lo)/max(hi-lo,1e-6),0,1)
            # match spatial size
            if s2.shape[1:] != s1.shape[1:]:
                H,W=s1.shape[1],s1.shape[2]
                s2 = np.stack([_cv2.resize(s2[c],(W,H),interpolation=_cv2.INTER_LINEAR) for c in range(s2.shape[0])])
            img = np.concatenate([s1,s2],axis=0)
            with st.spinner("Running sliding-window inference…"):
                mask,prob = predict(model,device,img,threshold)
            st.metric("Flooded Area",f"{mask.mean()*100:.1f}%")
            rgb = to_rgb(img)
            fig = overlay_fig(rgb,mask,title="Raw S1+S2")
            st.pyplot(fig,use_container_width=True); plt.close(fig)


# ── Evaluate ──────────────────────────────────────────────────────────────────

elif "Evaluate" in page:
    st.title("📊 Batch Evaluation")
    from config import PROC_DIR
    if not sel_ckpt: st.warning("Train first: `python train.py`"); st.stop()
    model,device,_ = load_model(sel_ckpt)

    img_dir  = PROC_DIR/"images"
    mask_dir = PROC_DIR/"masks"
    tiles = sorted(p.stem for p in img_dir.glob("*.tif")) if img_dir.exists() else []

    if not tiles:
        st.info("No tiles found.\n```\npython generate_sample_data.py --n_tiles 100\n```")
        st.stop()

    n_eval = st.slider("Tiles to evaluate",10,min(len(tiles),500),min(50,len(tiles)))
    if st.button("▶ Run Evaluation",type="primary"):
        from utils.metrics import MetricTracker
        tracker,rows,prog = MetricTracker(),[],st.progress(0,"Evaluating…")
        for i,tid in enumerate(tiles[:n_eval]):
            arr = tifffile.imread(str(img_dir/f"{tid}.tif")).astype(np.float32)
            if arr.ndim==2: arr=arr[np.newaxis]
            if arr.shape[-1]<arr.shape[0]: pass
            elif arr.ndim==3 and arr.shape[2]<arr.shape[0]: arr=np.transpose(arr,(2,0,1))
            gt_path = mask_dir/f"{tid}_mask.tif"
            if not gt_path.exists(): continue
            gt = (tifffile.imread(str(gt_path))>0).astype(np.uint8)
            if gt.ndim==3: gt=gt[0]
            pred_mask,_ = predict(model,device,arr,threshold)
            m = metrics(pred_mask,gt); m["tile"]=tid; m["flood_%"]=float(pred_mask.mean()*100)
            tracker.update({k:v for k,v in m.items() if k not in ("tile","flood_%")})
            rows.append(m)
            prog.progress((i+1)/n_eval,text=f"{i+1}/{n_eval} tiles")
        prog.empty()
        means = tracker.means()
        cols = st.columns(len(means))
        for col,(k,v) in zip(cols,means.items()):
            col.metric(k,f"{v:.4f}")
        df = pd.DataFrame(rows)
        st.dataframe(df,use_container_width=True)
        st.plotly_chart(
            px.histogram(df,x="IoU",nbins=20,template="plotly_dark",
                         color_discrete_sequence=["#2d8cf0"])
            .update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#161b22",height=280),
            use_container_width=True)
        st.download_button("⬇ CSV",df.to_csv(index=False),"evaluation.csv","text/csv")


# ── Logs ──────────────────────────────────────────────────────────────────────

elif "Logs" in page:
    st.title("📈 Training Logs")
    from config import LOG_DIR
    runs = sorted(LOG_DIR.glob("run_*"),reverse=True)
    if not runs:
        st.info("No runs yet.\n```\npython train.py\n```\nOr view in TensorBoard:\n```\ntensorboard --logdir logs/\n```")
        st.stop()
    sel_run = st.selectbox("Run",[d.name for d in runs])
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(LOG_DIR/sel_run)); ea.Reload()
        tags = ea.Tags().get("scalars",[])
        if not tags: st.info("Run still empty."); st.stop()
        import pandas as _pd
        records=[{"step":ev.step,"metric":t,"value":ev.value}
                 for t in tags for ev in ea.Scalars(t)]
        df = _pd.DataFrame(records)
        done = set()
        for tag in tags:
            base = tag.split("/")[0] if "/" in tag else tag
            if base in done: continue
            done.add(base)
            group = [t for t in tags if t.startswith(base)]
            st.markdown(f"#### {base}")
            fig=go.Figure()
            for t in group:
                sub=df[df.metric==t]
                fig.add_trace(go.Scatter(x=sub.step,y=sub.value,
                              mode="lines",name=t.split("/")[-1],line=dict(width=2)))
            fig.update_layout(template="plotly_dark",paper_bgcolor="#0d1117",
                              plot_bgcolor="#161b22",height=240,
                              margin=dict(t=10,b=30,l=40,r=20))
            st.plotly_chart(fig,use_container_width=True)
    except ImportError:
        st.warning("Install tensorboard for in-app log charts: `pip install tensorboard`\n\n"
                   "Or open directly: `tensorboard --logdir logs/`")


# ── About ──────────────────────────────────────────────────────────────────────

elif "About" in page:
    st.title("ℹ️ About")
    st.markdown("""
**AI-Based Flood Impact Assessment** — Binary semantic segmentation of flood-inundated
regions using fused Sentinel-1 SAR and Sentinel-2 optical imagery.

**Dataset:** [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) — 446 hand-labelled
scenes, 11 global flood events, 10 m resolution.

| Ch | Band | Description |
|----|------|-------------|
| 0 | S1 VV | SAR backscatter co-pol |
| 1 | S1 VH | SAR backscatter cross-pol |
| 2 | S2 B2 | Blue 490 nm |
| 3 | S2 B3 | Green 560 nm |
| 4 | S2 B4 | Red 665 nm |
| 5 | S2 B8 | NIR 842 nm |
| 6 | S2 B11 | SWIR1 1610 nm |

**Model:** U-Net · ResNet-34 encoder · Dice+BCE loss · ~24M parameters
""")
    import platform
    st.code(f"Python {platform.python_version()}  |  "
            f"PyTorch {torch.__version__}  |  "
            f"CUDA: {torch.cuda.is_available()}")
