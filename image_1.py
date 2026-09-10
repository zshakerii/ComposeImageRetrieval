import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

plt.rcParams["font.family"] = "DejaVu Sans"
BG = "#12141c"
PIPE = "#2f6fe4"; PIPE_F = "#1b3a73"
GREEN = "#35b45f"; GREEN_F = "#164a28"
ORANGE = "#f2994a"; ORANGE_F = "#7a4416"
GRAY = "#8b93a3"; GRAY_F = "#2e333d"
PURPLE = "#9b6ef3"; PURPLE_F = "#3c2a6b"
TEXT = "#e8eaf0"

fig, ax = plt.subplots(figsize=(16,12), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0,160); ax.set_ylim(0,120); ax.axis("off")

def box(x,y,w,h,label,fc,ec,fs=11,sub=None):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",
                 fc=fc, ec=ec, lw=1.8, zorder=2))
    dy = 1.2 if sub else 0
    ax.text(x+w/2, y+h/2+dy, label, ha="center", va="center",
            color=TEXT, fontsize=fs, fontweight="bold", zorder=3)
    if sub:
        ax.text(x+w/2, y+h/2-1.8, sub, ha="center", va="center",
                color="#c8cdd8", fontsize=8, zorder=3)

def arrow(x1,y1,x2,y2,color=PIPE,ls="-",lw=1.6,rad=0.0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>", mutation_scale=14,
                 color=color, lw=lw, linestyle=ls, zorder=1,
                 connectionstyle=f"arc3,rad={rad}"))

ax.text(80,116,"Composed Image Retrieval Harness — Architecture", ha="center", color=TEXT, fontsize=19, fontweight="bold")
ax.text(80,112.7,"pasted-text-(3).txt  •  CIRR / CIRCO zero-shot evaluation", ha="center", color=GRAY, fontsize=10)

box(60,104,40,6,"main()  —  orchestration",PIPE_F,PIPE,fs=13)
box(48,94,64,7,"Argument Parsing Layer  (argparse)",PIPE_F,PIPE,fs=11.5,
    sub="9 models • paths • k-values (1,5,10,50) • protocols • cache options")
arrow(80,104,80,101.2)

box(30,82,52,8,"Data / GT Loading Pipeline",PIPE_F,PIPE,fs=11.5,
    sub="JSON → parse_cirr/parse_circo → GT attach → caption fallback")
arrow(80,94,80,90.4)
box(92,82,38,8,"Image Handling",GRAY_F,GRAY,fs=10.5,
    sub="build_image_path_index() • scan_gallery_ids() • load_image()")
arrow(82,86,92,86,color=GRAY)

ax.text(80,79.2,"Model Backends (9)", ha="center", color=GREEN, fontsize=11.5, fontweight="bold")
models = [("CLIP","ViT-B/32 • shared backbone"),("OpenCLIP","multiple checkpoints"),("BLIP","image-text matching"),
          ("SearLE","CLIP-based feature combiner"),("SigLIP","sigmoid loss"),("ALIGN","dual-encoder"),
          ("Qwen3","Qwen3-VL embedding"),("CLIP-Beta","fine-tuned variant"),("Adaptive-RRF","4-channel fusion ★")]
mx, my, mw, mh, gx, gy = 22, 56, 34, 13, 6.5, 5.5
mpos = {}
for i,(name,sub) in enumerate(models):
    r,c = divmod(i,3)
    x = mx + c*(mw+gx); y = my - r*(mh+gy)
    fc, ec = (ORANGE_F,ORANGE) if name=="Adaptive-RRF" else (GREEN_F,GREEN)
    box(x,y,mw,mh,name,fc,ec,fs=11,sub=sub)
    mpos[name]=(x,y)

box(22,38,50,7,"Shared CLIP Backbone",GRAY_F,GRAY,fs=10.5)
arrow(mpos["CLIP"][0]+mw/2, mpos["CLIP"][1], 47, 45.2, color=GRAY)
arrow(mpos["SearLE"][0]+mw/2, mpos["SearLE"][1], 47, 45.2, color=GRAY)
box(84,38,50,7,"Shared CLIP Backbone → CLIP-Beta / Adaptive-RRF",GRAY_F,GRAY,fs=10.5)
arrow(mpos["CLIP-Beta"][0]+mw/2, mpos["CLIP-Beta"][1], 109, 45.2, color=GRAY)
arrow(mpos["Adaptive-RRF"][0]+mw/2, mpos["Adaptive-RRF"][1], 109, 45.2, color=GRAY)
arrow(72,41.5,84,41.5,color=GRAY)

box(8,26,52,7,"Cache System  •  *.pt files",GRAY_F,GRAY,fs=10,
    sub="embedding_cache/<dataset>/<split>/ • v4 payload • atomic save")
arrow(30,38,34,33.2,color=GRAY)

ax.text(112,36.5,"Adaptive-RRF Sub-module", ha="center", color=ORANGE, fontsize=11, fontweight="bold")
chans = [("base","CLIP base score"),("ref","reference/caption"),("qwen","Qwen3 channel"),("region","region crop")]
cw,chh = 21,4.6
for i,(n,s) in enumerate(chans):
    x = 84 + i*(cw+2.2)
    box(x,29,cw,chh,n,ORANGE_F,ORANGE,fs=9.5)
    ax.text(x+cw/2, 27.4, s, ha="center", va="center", color="#c8cdd8", fontsize=7)
    arrow(x+cw/2, 29, 124, 25.8, color=ORANGE, rad=0.1)
box(101,19.5,46,6,"RRF Fusion  →  final score",ORANGE_F,ORANGE,fs=10.5,
    sub="score = Σ 1/(k + rank_channel)")
arrow(124,25.8,124,25.5,color=ORANGE)

box(30,8,100,8,"Evaluation / Metrics Layer",PURPLE_F,PURPLE,fs=12,
    sub="Recall@K (1,5,10,50) • mAP@{5,10,50} • MRR   |   protocols: full_gallery • cirr_subset (rank members only)")
arrow(109,19.5,109,16.2,color=ORANGE,rad=0.08)
arrow(47,26,55,16.2,color=GRAY,rad=0.1)

box(50,0.5,60,6,"Output: results dict → summarize_results() → LaTeX table / JSON",PURPLE_F,PURPLE,fs=10)
arrow(80,8,80,6.7,color=PURPLE)

handles = [mlines.Line2D([],[],color=c,lw=3,label=l) for c,l in
           [(PIPE,"Core pipeline"),(GREEN,"Model backends"),(ORANGE,"Adaptive-RRF"),(GRAY,"Cache / storage"),(PURPLE,"Metrics / output")]]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005,0.995),
          frameon=True, facecolor="#1a1d27", edgecolor="#3a3f4c", labelcolor=TEXT, fontsize=9)

plt.savefig("/mnt/data/harness_architecture.png", dpi=200, bbox_inches="tight", facecolor=BG)
import os
from PIL import Image
im = Image.open("/mnt/data/harness_architecture.png")
print("saved:", os.path.getsize("/mnt/data/harness_architecture.png"), "bytes, size:", im.size)