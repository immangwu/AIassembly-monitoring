import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
import numpy as np
import os, time, base64, io
import requests
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Custom camera component — works on Meta Quest + standard browsers
_camera_xr = components.declare_component(
    "camera_xr",
    path=os.path.join(BASE_DIR, "camera_component")
)

def camera_xr(key=None):
    """Returns base64 JPEG data-URL string when user captures, else None."""
    return _camera_xr(key=key, default=None)

st.set_page_config(page_title="TANCAM Valve Assembly", page_icon="🔧", layout="wide",
                   initial_sidebar_state="collapsed")

# ──────────────────────────────────────────────────────────
# ONNX MODEL  (cached — loads once, no PyTorch needed)
# ──────────────────────────────────────────────────────────
INPUT_SIZE  = 640
NUM_CLASSES = 5
PART_NAMES  = {0: "part 1", 1: "part 2", 2: "part 3", 3: "part 4", 4: "part 5"}
CONF_THRESH = 0.05
IOU_THRESH  = 0.45

@st.cache_resource
def load_model():
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(
        os.path.join(BASE_DIR, "best.onnx"),
        sess_options=opts,
        providers=["CPUExecutionProvider"]
    )

def _preprocess(pil_img):
    orig_w, orig_h = pil_img.size
    img = pil_img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]   # (1,3,640,640)
    return arr, orig_w, orig_h

def _nms(boxes, scores):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        iou = (np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)) / \
              (areas[i]+areas[order[1:]]-np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)+1e-6)
        order = order[1:][iou <= IOU_THRESH]
    return keep

def _infer(pil_img):
    """Run ONNX inference. Returns list of {name, conf, x1,y1,x2,y2}."""
    session = load_model()
    inp, orig_w, orig_h = _preprocess(pil_img)
    pred = session.run(None, {session.get_inputs()[0].name: inp})[0][0]  # (41,8400)
    pred = pred.T                      # (8400,41)
    boxes_xywh    = pred[:, :4]
    class_scores  = pred[:, 4:4+NUM_CLASSES]
    class_ids     = np.argmax(class_scores, axis=1)
    confidences   = class_scores[np.arange(len(class_scores)), class_ids]
    mask          = confidences >= CONF_THRESH
    boxes_xywh, confidences, class_ids = boxes_xywh[mask], confidences[mask], class_ids[mask]
    if len(boxes_xywh) == 0:
        return []
    # cx,cy,w,h → x1,y1,x2,y2 scaled to original image
    bx = np.stack([
        (boxes_xywh[:,0]-boxes_xywh[:,2]/2)/INPUT_SIZE*orig_w,
        (boxes_xywh[:,1]-boxes_xywh[:,3]/2)/INPUT_SIZE*orig_h,
        (boxes_xywh[:,0]+boxes_xywh[:,2]/2)/INPUT_SIZE*orig_w,
        (boxes_xywh[:,1]+boxes_xywh[:,3]/2)/INPUT_SIZE*orig_h,
    ], axis=1)
    dets = []
    for cls_id in range(NUM_CLASSES):
        m = class_ids == cls_id
        if not m.any(): continue
        for k in _nms(bx[m], confidences[m]):
            b = bx[m][k]
            dets.append({"cls": cls_id, "name": PART_NAMES[cls_id],
                         "conf": float(confidences[m][k]),
                         "x1": int(b[0]), "y1": int(b[1]),
                         "x2": int(b[2]), "y2": int(b[3])})
    return dets

# ──────────────────────────────────────────────────────────
# STAGE DATA
# ──────────────────────────────────────────────────────────
STAGES = [
    dict(
        id=1, label="STAGE 1", short="P2 → P1",
        desc="Pick Part 2 (Hollow Ring) and mount it on Part 1 (Main Body)",
        parts=[("P2","Hollow Ring (Knurled)","Pick first — knurled surface",True),
               ("P1","Main Body + Handle","Base component — hold firmly",False)],
        need=["part 2","part 1"], wrong=["part 3","part 4","part 5"],
        result="—", video="VIDEO.mp4", is_final=False,
        torque="2 Nm", torque_note="Hand-tight fit — do not over-tighten",
        pos_length="15 mm", pos_note="Part 2 seated flush on Part 1 bore",
        instructions=[
            "Pick up Part 2 — the hollow ring with knurled exterior.",
            "Orient Part 1 (main body) with handle facing upward.",
            "Slide Part 2 into the cylindrical bore of Part 1.",
            "Align the holes on Part 2 with the holes on Part 1.",
            "Hold the assembly in front of the camera.",
            "Click ✅ Verify Stage when detection looks correct.",
        ],
        success="Part 2 aligned on Part 1 → Assembly A1 formed ✓"
    ),
    dict(
        id=2, label="STAGE 2", short="A1 + P3",
        desc="Hold Assembly A1 in hand and place End Cap Disc (Part 3) onto it",
        parts=[("A1","Assembly 1 (in hand)","Result from Stage 1",False),
               ("P3","End Cap Disc","Place on top of A1",True)],
        need=["part 3"], wrong=["part 4","part 5"],
        result="A1", video="VIDEO1.mp4", is_final=False,
        torque="3 Nm", torque_note="Press-fit — firm hand pressure only",
        pos_length="8 mm", pos_note="Part 3 disc seated fully into A1 end",
        instructions=[
            "Hold A1 (P1+P2) firmly in your non-dominant hand.",
            "Pick Part 3 — the circular disc / end cap.",
            "Note: two small nubs on Part 3 must face outward.",
            "Press Part 3 firmly into the open end of A1.",
            "You should feel or hear a click when seated correctly.",
            "Click ✅ Verify Stage when detection looks correct.",
        ],
        success="Part 3 seated on A1 → Assembly A2 formed ✓"
    ),
    dict(
        id=3, label="STAGE 3", short="A2 + P4",
        desc="Hold Assembly A2 and insert the Long Needle Pin (Part 4)",
        parts=[("A2","Assembly 2 (in hand)","Result from Stage 2",False),
               ("P4","Long Needle Pin","Thread through the hole",True)],
        need=["part 4"], wrong=["part 5"],
        result="A2", video="VIDEO2.mp4", is_final=False,
        torque="1.5 Nm", torque_note="Slide-fit — equal protrusion both sides",
        pos_length="Equal protrusion", pos_note="Pin centred ± 1 mm through A2 body",
        instructions=[
            "Hold A2 firmly — keep the end-cap face accessible.",
            "Pick Part 4 — the long thin pin with a small hole at one end.",
            "Locate the through-hole in A2's body.",
            "Thread Part 4 through the hole — small hole end first.",
            "Push until Part 4 protrudes equally on both sides.",
            "Click ✅ Verify Stage when detection looks correct.",
        ],
        success="Part 4 threaded in A2 → Assembly A3 formed ✓"
    ),
    dict(
        id=4, label="STAGE 4", short="A3 + P5",
        desc="Hold Assembly A3 and insert the Small Peg/Dowel (Part 5) to lock",
        parts=[("A3","Assembly 3 (in hand)","Result from Stage 3",False),
               ("P5","Small Peg (Dowel)","Locks the needle pin in place",True)],
        need=["part 5"], wrong=[],
        result="A3", video="VIDEO3.mp4", is_final=False,
        torque="1.5 Nm", torque_note="Interference fit — push until flush",
        pos_length="Flush fit", pos_note="Peg flush with outer surface of A3",
        instructions=[
            "Hold A3 steady — Part 4 should be fully inserted.",
            "Pick Part 5 — the short cylindrical dowel / peg.",
            "Locate the small cross-hole near the end of Part 4.",
            "Push Part 5 through this cross-hole to lock Part 4.",
            "Part 5 acts as a retaining pin.",
            "Click ✅ Verify Stage when detection looks correct.",
        ],
        success="Part 5 locked in A3 → Assembly A4 formed ✓"
    ),
    dict(
        id=5, label="STAGE 5", short="Torque Check",
        desc="Final torque + positioning check — apply 15 Nm and hold A4 ≤ 10 cm from sensor",
        parts=[("A4","Full Assembly (A4)","Apply final torque with wrench",True)],
        need=[], wrong=[],
        result="A4", video=None, is_final=True,
        torque="15 Nm", torque_note="Use calibrated torque wrench — do not exceed 18 Nm",
        pos_length="≤ 10 cm", pos_note="Assembly within sensor range for QC confirmation",
        instructions=[
            "Hold the completed assembly (A4) firmly.",
            "Use calibrated torque wrench — apply exactly 15 Nm.",
            "Do NOT exceed 18 Nm (risk of thread damage).",
            "Hold the assembly in front of the ultrasonic sensor (≤ 10 cm).",
            "Click 📡 Read Sensor to confirm positioning.",
            "Click ✅ COMPLETE ASSEMBLY when all checks pass.",
        ],
        success="Torque 15 Nm applied · Positioning confirmed — A4 complete! ✓"
    ),
]

# ──────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────
def init_state():
    defaults = dict(screen="home", stage=0, done=[], start_time=None,
                    det_result=None, annotated=None, stage_ok=False,
                    sensor_dist=50.0, sensor_source=None, nodemcu_ip="192.168.1.100",
                    resp_time=None, stage_times={}, part_distances=[])
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ──────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important;}
div.block-container{padding:0.5rem 1.2rem 1rem;}
body,[data-testid="stAppViewContainer"]{background:#0a0c10;}

/* Header */
.hdr{background:linear-gradient(135deg,#0d1117,#111827);border-bottom:2px solid #f59e0b;
     padding:10px 20px;display:flex;align-items:center;justify-content:space-between;
     margin:-0.5rem -1.2rem 1.2rem;box-shadow:0 4px 30px rgba(245,158,11,0.12);}
.hdr-hex{width:36px;height:36px;background:linear-gradient(135deg,#f59e0b,#d97706);
         clip-path:polygon(50% 0%,93% 25%,93% 75%,50% 100%,7% 75%,7% 25%);
         display:inline-flex;align-items:center;justify-content:center;
         font-weight:700;font-size:11px;color:#000;margin-right:12px;vertical-align:middle;}
.hdr-title{font-size:17px;font-weight:700;color:#f59e0b;letter-spacing:2px;}
.hdr-sub{font-size:10px;color:#64748b;letter-spacing:1px;}
.hdr-badge{font-family:monospace;font-size:11px;background:rgba(245,158,11,0.1);
           border:1px solid rgba(245,158,11,0.3);color:#f59e0b;padding:4px 12px;border-radius:3px;}

/* Stage items */
.st-item{display:flex;align-items:center;gap:10px;padding:9px 11px;border-radius:7px;margin-bottom:5px;}
.st-done{background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);}
.st-active{background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.4);}
.st-pending{background:#111419;border:1px solid #1e2530;opacity:0.45;}
.st-icon{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
         justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;}
.ic-done{background:rgba(16,185,129,0.2);color:#10b981;border:1px solid #10b981;}
.ic-active{background:rgba(245,158,11,0.2);color:#f59e0b;border:1px solid #f59e0b;}
.ic-pending{background:#181c23;color:#64748b;border:1px solid #1e2530;}
.st-lbl{font-size:12px;font-weight:600;color:#e2e8f0;}
.st-sub{font-size:10px;color:#64748b;}

/* Part pills */
.pp{display:flex;align-items:center;gap:10px;padding:8px 11px;border-radius:6px;margin-bottom:5px;}
.pp-hi{background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.4);}
.pp-nm{background:#181c23;border:1px solid #1e2530;}
.pp-num{font-size:20px;font-weight:700;color:#f59e0b;font-family:monospace;min-width:28px;}
.pp-name{font-size:12px;font-weight:600;color:#e2e8f0;}
.pp-note{font-size:10px;color:#64748b;}

/* Result items */
.ri{border-radius:6px;padding:8px 12px;margin-bottom:6px;}
.ri-ok  {background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);}
.ri-miss{background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);}
.ri-ext {background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);}
.ri-t{font-weight:700;font-size:13px;}
.ri-s{font-size:11px;opacity:0.7;margin-top:2px;}

/* Section title */
.sec-title{font-size:10px;letter-spacing:3px;color:#64748b;text-transform:uppercase;
           padding-bottom:6px;border-bottom:1px solid #1e2530;margin-bottom:9px;}
/* Instruction step */
.is{padding:4px 0;font-size:12px;color:#94a3b8;border-bottom:1px solid #1e2530;}

/* Stage label */
.stage-lbl{font-size:26px;font-weight:800;color:#f59e0b;letter-spacing:1px;}
.stage-desc{color:#64748b;font-size:12px;margin-top:2px;}
.asm-code{font-size:22px;font-weight:700;color:#10b981;font-family:monospace;text-align:right;}
.asm-sub{font-size:10px;color:#64748b;text-align:right;}

/* Home title */
.home-title{font-size:clamp(26px,4vw,52px);font-weight:800;text-align:center;
            background:linear-gradient(135deg,#f59e0b,#fbbf24,#d97706);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.15;}
.home-sub{text-align:center;color:#64748b;font-size:12px;letter-spacing:3px;margin-top:6px;}
.pc{background:#111419;border:1px solid #1e2530;border-radius:8px;
    padding:16px 10px;text-align:center;}
.pc-num{font-size:26px;font-weight:700;color:#f59e0b;font-family:monospace;}
.pc-nm{font-size:10px;color:#64748b;margin-top:4px;line-height:1.4;}

/* Live badge */
.live-badge{display:inline-block;background:#ef4444;color:#fff;font-size:10px;
            font-weight:700;padding:2px 8px;border-radius:3px;letter-spacing:2px;
            animation:blink 1.2s step-start infinite;}
@keyframes blink{50%{opacity:0.3;}}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# HEADER (always visible)
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div>
    <span class="hdr-hex">TN</span>
    <span class="hdr-title">VALVE ASSEMBLY SYSTEM</span><br>
    <span style="margin-left:48px;" class="hdr-sub">TN-IMPACT · TNI26175 · VALVE ASSEMBLY &amp; QUALITY CONTROL</span>
  </div>
  <div class="hdr-badge">14-03-2026</div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# YOLO ANNOTATION HELPER  (used by live processor + verify)
# ──────────────────────────────────────────────────────────
def _dashed_line(draw, p1, p2, fill, width=2, dash=10, gap=5):
    """Draw a dashed line between two points."""
    x1, y1 = p1;  x2, y2 = p2
    length = max(((x2-x1)**2 + (y2-y1)**2) ** 0.5, 1)
    dx, dy = (x2-x1)/length, (y2-y1)/length
    pos = 0
    while pos < length:
        sx = x1 + dx * pos;  sy = y1 + dy * pos
        ex = x1 + dx * min(pos+dash, length)
        ey = y1 + dy * min(pos+dash, length)
        draw.line([(int(sx), int(sy)), (int(ex), int(ey))], fill=fill, width=width)
        pos += dash + gap


def annotate_frame(pil_img: Image.Image, stage_idx: int):
    """
    Annotate PIL image with YOLO boxes + inter-part distance lines.
    Returns (annotated PIL, detections list, part_distances list).
    """
    raw_dets = _infer(pil_img)

    img_out = pil_img.convert("RGB").copy()
    draw    = ImageDraw.Draw(img_out)
    stage   = STAGES[stage_idx]
    detections = []
    centers    = {}   # name → (cx, cy)

    for det in raw_dets:
        name  = det["name"]
        conf  = det["conf"]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

        needed = name in stage["need"]
        wrong  = name in stage["wrong"]
        color  = (239,68,68) if wrong else (16,185,129) if needed else (245,158,11)

        for th in range(3):
            draw.rectangle([x1-th, y1-th, x2+th, y2+th], outline=color, width=1)

        cs = 16
        for (ccx, ccy, ddx, ddy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            draw.line([(ccx,ccy),(ccx+ddx*cs,ccy)], fill=color, width=3)
            draw.line([(ccx,ccy),(ccx,ccy+ddy*cs)], fill=color, width=3)

        icon  = "✗" if wrong else "✓"
        label = f"{icon} {name.upper()}  {round(conf*100)}%"
        tw    = len(label) * 7
        ly    = y1 - 20 if y1 > 24 else y2 + 4
        draw.rectangle([x1, ly, x1+tw+8, ly+18], fill=color)
        draw.text((x1+4, ly+2), label, fill=(0,0,0))

        bcx, bcy = (x1+x2)//2, (y1+y2)//2
        draw.line([(bcx-6, bcy), (bcx+6, bcy)], fill=color, width=2)
        draw.line([(bcx, bcy-6), (bcx, bcy+6)], fill=color, width=2)

        centers[name] = (bcx, bcy)
        detections.append({"name": name, "conf": conf,
                            "needed": needed, "wrong": wrong,
                            "cx": bcx, "cy": bcy,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    # ── Draw distance lines between every detected pair ──────
    part_distances = []
    names = list(centers.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2   = names[i], names[j]
            cx1, cy1 = centers[n1]
            cx2, cy2 = centers[n2]
            dist_px  = int(((cx2-cx1)**2 + (cy2-cy1)**2) ** 0.5)

            # Color: cyan if both are needed, white otherwise
            both_needed = (n1 in stage["need"] and n2 in stage["need"])
            line_color  = (0, 220, 255) if both_needed else (200, 200, 200)

            # Dashed line between centres
            _dashed_line(draw, (cx1, cy1), (cx2, cy2), fill=line_color, width=2)

            # Distance badge at midpoint
            mx, my   = (cx1+cx2)//2, (cy1+cy2)//2
            dist_lbl = f"d={dist_px}px"
            bw       = len(dist_lbl)*7 + 8
            draw.rectangle([mx-bw//2, my-10, mx+bw//2, my+10],
                           fill=(20, 20, 20), outline=line_color)
            draw.text((mx-bw//2+4, my-8), dist_lbl, fill=line_color)

            part_distances.append({
                "parts":       (n1, n2),
                "distance_px": dist_px,
                "both_needed": both_needed,
            })

    return img_out, detections, part_distances


def analyse(detections, stage_idx):
    found   = list({d["name"] for d in detections})
    stage   = STAGES[stage_idx]
    missing = [p for p in stage["need"] if p not in found]
    extra   = [p for p in found if p in stage["wrong"]]
    correct = [p for p in stage["need"] if p in found]
    neutral = [p for p in found if p not in stage["need"] and p not in stage["wrong"] and stage["need"]]
    return missing, extra, correct, neutral




# ══════════════════════════════════════════════════════════
# SCREEN: HOME
# ══════════════════════════════════════════════════════════
if st.session_state.screen == "home":
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="home-title">VALVE ASSEMBLY<br>GUIDANCE SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-sub">5-PART ASSEMBLY · LIVE YOLO DETECTION · POKA-YOKE ENABLED</div>',
                unsafe_allow_html=True)
    st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)

    cols = st.columns(5)
    parts_home = [("P1","Main Body\n+ Handle"),("P2","Hollow Ring\n(Knurled)"),
                  ("P3","End Cap\n(Disc)"),("P4","Long Needle\nPin"),("P5","Small Peg\n(Dowel)")]
    for col, (num, name) in zip(cols, parts_home):
        col.markdown(f'<div class="pc"><div class="pc-num">{num}</div>'
                     f'<div class="pc-nm">{name.replace(chr(10),"<br>")}</div></div>',
                     unsafe_allow_html=True)

    st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)
    _, mc, _ = st.columns([1,2,1])
    with mc:
        if st.button("▶   BEGIN ASSEMBLY", use_container_width=True, type="primary"):
            st.session_state.update(screen="assembly", stage=0, done=[],
                                    start_time=time.time(), det_result=None,
                                    annotated=None, stage_ok=False, sensor_dist=50.0)
            st.rerun()


# ══════════════════════════════════════════════════════════
# SCREEN: ASSEMBLY
# ══════════════════════════════════════════════════════════
elif st.session_state.screen == "assembly":
    s   = STAGES[st.session_state.stage]
    L, C, R = st.columns([1, 2.1, 1.1])

    # ── LEFT: Progress sidebar ────────────────────────────
    with L:
        st.markdown('<div class="sec-title">Assembly Progress</div>', unsafe_allow_html=True)
        for i, sd in enumerate(STAGES):
            if i in st.session_state.done:
                ic, sc = "ic-done",   "st-done";   check = "✓"
            elif i == st.session_state.stage:
                ic, sc = "ic-active", "st-active"; check = str(sd["id"])
            else:
                ic, sc = "ic-pending","st-pending"; check = str(sd["id"])
            st.markdown(
                f'<div class="st-item {sc}">'
                f'<div class="st-icon {ic}">{check}</div>'
                f'<div><div class="st-lbl">{sd["short"]}</div>'
                f'<div class="st-sub">{sd["label"]}</div></div></div>',
                unsafe_allow_html=True)

        pct = len(st.session_state.done) / len(STAGES)
        st.progress(pct, text=f"{len(st.session_state.done)} / {len(STAGES)} stages")

    # ── CENTER: Live camera + Detection ───────────────────
    with C:
        ch1, ch2 = st.columns([2,1])
        with ch1:
            st.markdown(f'<div class="stage-lbl">{s["label"]}</div>'
                        f'<div class="stage-desc">{s["desc"]}</div>',
                        unsafe_allow_html=True)
        with ch2:
            st.markdown(f'<div class="asm-code">{s["result"]}</div>'
                        f'<div class="asm-sub">CURRENT ASSEMBLY</div>',
                        unsafe_allow_html=True)

        st.divider()

        if not s["is_final"]:

            # ── CAMERA CAPTURE ──────────────────────────
            st.markdown('<span style="background:#1d4ed8;color:#fff;font-size:10px;'
                        'font-weight:700;padding:2px 8px;border-radius:3px;letter-spacing:2px;">'
                        '⬤ CAPTURE</span>'
                        '<span style="color:#64748b;font-size:11px;margin-left:8px;">'
                        'Take a photo — YOLO will annotate automatically</span>',
                        unsafe_allow_html=True)

            cam_data = camera_xr(key=f"cam_stage_{st.session_state.stage}")

            if cam_data:
                t_start = time.time()
                # Decode base64 data-URL → PIL image
                header, b64 = cam_data.split(",", 1)
                pil_img = Image.open(io.BytesIO(base64.b64decode(b64)))
                annotated, detections, part_distances = annotate_frame(pil_img, st.session_state.stage)
                missing, extra, correct, neutral = analyse(detections, st.session_state.stage)
                st.session_state.annotated      = annotated
                st.session_state.part_distances = part_distances
                st.session_state.resp_time      = round(time.time() - t_start, 3)
                st.session_state.det_result     = dict(
                    missing=missing, extra=extra,
                    correct=correct, neutral=neutral,
                    detections=detections
                )
                st.session_state.stage_ok = (not missing and not extra and bool(correct))

            # Show snapshot after verify
            if st.session_state.annotated:
                st.image(st.session_state.annotated,
                         caption="Verified snapshot — green=correct  red=wrong  orange=not needed",
                         use_container_width=True)

            # Response time + result panel
            if st.session_state.det_result:
                res = st.session_state.det_result

                # ── Response time row ──
                if st.session_state.resp_time is not None:
                    rt = st.session_state.resp_time
                    rt_color = "#10b981" if rt < 1.0 else "#f59e0b" if rt < 3.0 else "#ef4444"
                    st.markdown(
                        f'<div style="display:flex;gap:16px;margin-bottom:8px;">'
                        f'<div style="background:#111419;border:1px solid #1e2530;border-radius:6px;'
                        f'padding:6px 14px;font-size:11px;color:#64748b;">'
                        f'⏱ Response Time: <span style="color:{rt_color};font-weight:700;">{rt*1000:.0f} ms</span></div>'
                        f'<div style="background:#111419;border:1px solid #1e2530;border-radius:6px;'
                        f'padding:6px 14px;font-size:11px;color:#64748b;">'
                        f'🔍 Detections: <span style="color:#e2e8f0;font-weight:700;">{len(res["detections"])}</span></div>'
                        f'</div>',
                        unsafe_allow_html=True)

                st.markdown("**Verification Result**")

                if not res["detections"]:
                    st.error("🔍 No parts detected — ensure parts are clearly visible and well-lit, then verify again.")
                else:
                    for p in res["correct"]:
                        d = next((x for x in res["detections"] if x["name"]==p), {})
                        st.markdown(
                            f'<div class="ri ri-ok">'
                            f'<div class="ri-t">✓ {p.upper()} — CORRECT</div>'
                            f'<div class="ri-s">Confidence: {round(d.get("conf",0)*100)}%</div></div>',
                            unsafe_allow_html=True)
                    for p in res["missing"]:
                        st.markdown(
                            f'<div class="ri ri-miss">'
                            f'<div class="ri-t">✗ {p.upper()} — MISSING</div>'
                            f'<div class="ri-s">Required for Stage {st.session_state.stage+1} — place it in view and verify again</div></div>',
                            unsafe_allow_html=True)
                    for p in res["extra"]:
                        d = next((x for x in res["detections"] if x["name"]==p), {})
                        st.markdown(
                            f'<div class="ri ri-ext">'
                            f'<div class="ri-t">⚠ {p.upper()} — WRONG PART</div>'
                            f'<div class="ri-s">Remove this — not needed at Stage {st.session_state.stage+1}. Conf: {round(d.get("conf",0)*100)}%</div></div>',
                            unsafe_allow_html=True)
                    for p in res["neutral"]:
                        d = next((x for x in res["detections"] if x["name"]==p), {})
                        st.markdown(
                            f'<div class="ri ri-ext">'
                            f'<div class="ri-t">⚠ {p.upper()} — NOT NEEDED NOW</div>'
                            f'<div class="ri-s">Set aside for a later stage. Conf: {round(d.get("conf",0)*100)}%</div></div>',
                            unsafe_allow_html=True)

            # Positioning distance panel
            if st.session_state.part_distances:
                st.markdown('<div class="sec-title" style="margin-top:10px;">Positioning Length (YOLO)</div>',
                            unsafe_allow_html=True)
                for pd in st.session_state.part_distances:
                    n1, n2   = pd["parts"]
                    dist_px  = pd["distance_px"]
                    color_cls = "ri-ok" if pd["both_needed"] else "ri-ext"
                    tag       = "Key pair" if pd["both_needed"] else "Pair"
                    st.markdown(
                        f'<div class="ri {color_cls}">'
                        f'<div class="ri-t">📏 {n1.upper()} ↔ {n2.upper()}</div>'
                        f'<div class="ri-s">{tag} · Centre-to-centre distance: '
                        f'<span style="font-weight:700;color:#e2e8f0;">{dist_px} px</span>'
                        f'</div></div>',
                        unsafe_allow_html=True)

            # Next stage button
            if st.session_state.stage_ok:
                st.success(f"✅  {s['success']}")
                if st.button("NEXT STAGE  →", use_container_width=True, type="primary"):
                    st.session_state.done.append(st.session_state.stage)
                    nxt = st.session_state.stage + 1
                    st.session_state.update(stage=nxt, det_result=None,
                                            annotated=None, stage_ok=False)
                    if nxt >= len(STAGES):
                        st.session_state.screen = "complete"
                    st.rerun()

            # Manual override — in case model misclassifies but parts are physically correct
            with st.expander("⚠ Model wrong? Override detection"):
                st.warning("Use this only if you have physically verified the correct parts are present and the model is misidentifying them.")
                if st.button("✔ I have the correct parts — Proceed to Next Stage",
                             use_container_width=True):
                    st.session_state.done.append(st.session_state.stage)
                    nxt = st.session_state.stage + 1
                    st.session_state.update(stage=nxt, det_result=None,
                                            annotated=None, stage_ok=False)
                    if nxt >= len(STAGES):
                        st.session_state.screen = "complete"
                    st.rerun()

        else:
            # ── STAGE 5: Torque + Positioning (NodeMCU) ─
            st.markdown(
                '<div style="background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.3);'
                'border-radius:8px;padding:10px 14px;margin-bottom:12px;">'
                '<span style="color:#f59e0b;font-weight:700;">STAGE 5 CHECKS</span>'
                '<span style="color:#94a3b8;font-size:12px;margin-left:10px;">'
                '① Apply 15 Nm torque  ② Hold assembly ≤ 10 cm from ultrasonic sensor</span></div>',
                unsafe_allow_html=True)

            # ── Torque entry ────────────────────────────
            st.markdown('<div class="sec-title">Torque Verification</div>', unsafe_allow_html=True)
            t_col1, t_col2 = st.columns([2, 1])
            with t_col1:
                torque_val = st.number_input(
                    "Applied Torque (Nm)",
                    min_value=0.0, max_value=30.0, step=0.5,
                    value=float(st.session_state.get("torque_applied", 0.0)),
                    help="Enter the torque value shown on your wrench after tightening."
                )
                st.session_state.torque_applied = torque_val
            with t_col2:
                target_nm = 15.0
                if torque_val == 0:
                    st.metric("Target", "15 Nm", delta="Enter value above")
                elif 13.0 <= torque_val <= 17.0:
                    st.metric("Torque Status", f"{torque_val} Nm", delta="✓ Within spec")
                elif torque_val > 17.0:
                    st.metric("Torque Status", f"{torque_val} Nm", delta="⚠ Over-torque!", delta_color="inverse")
                else:
                    st.metric("Torque Status", f"{torque_val} Nm", delta="Under target", delta_color="inverse")

            torque_ok = 13.0 <= torque_val <= 17.0
            if torque_val > 0 and not torque_ok:
                if torque_val > 17.0:
                    st.error(f"⚠ Over-torque ({torque_val} Nm) — risk of thread damage! Max allowed: 18 Nm. Loosen and re-torque.")
                else:
                    st.warning(f"⚠ Torque {torque_val} Nm is below target 15 Nm — apply more torque.")
            elif torque_ok:
                st.success(f"✅ Torque {torque_val} Nm — within specification (13–17 Nm)")

            st.divider()

            # ── Positioning Length (NodeMCU HC-SR04) ───
            st.markdown('<div class="sec-title">Positioning Length (HC-SR04 Sensor)</div>',
                        unsafe_allow_html=True)

            ip_col, btn_col = st.columns([3, 1])
            with ip_col:
                st.session_state.nodemcu_ip = st.text_input(
                    "NodeMCU IP Address",
                    value=st.session_state.nodemcu_ip,
                    placeholder="e.g. 192.168.1.100",
                    help="Open Arduino Serial Monitor at 115200 baud — IP is printed on startup."
                )
            with btn_col:
                st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
                read_btn = st.button("📡 Read Sensor", use_container_width=True, type="primary")

            # Fetch positioning length from NodeMCU
            sensor_resp_time = None
            if read_btn:
                try:
                    t_s = time.time()
                    url  = f"http://{st.session_state.nodemcu_ip.strip()}/distance"
                    resp = requests.get(url, timeout=3)
                    sensor_resp_time = round((time.time() - t_s) * 1000)
                    data    = resp.json()
                    fetched = float(data.get("distance", -1))
                    if fetched < 0:
                        st.warning("⚠ Sensor timeout — object out of range (> 4 m). Move assembly closer.")
                    else:
                        st.session_state.sensor_dist   = fetched
                        st.session_state.sensor_source = "nodemcu"
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot reach NodeMCU at {st.session_state.nodemcu_ip} — check IP and WiFi.")
                except requests.exceptions.Timeout:
                    st.error("❌ NodeMCU did not respond in 3 s — check wiring and power.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

            dist   = st.session_state.sensor_dist
            source = st.session_state.get("sensor_source")

            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Positioning Length",
                          f"{dist:.1f} cm" if source else "— cm",
                          delta="✓ In position" if (source and dist <= 10) else "Move closer ≤ 10 cm")
            with m2:
                st.metric("Target Length", "≤ 10 cm", delta="Sensor threshold")
            with m3:
                rt_label = f"{sensor_resp_time} ms" if sensor_resp_time else ("NodeMCU HC-SR04" if source == "nodemcu" else "Simulation" if source else "—")
                st.metric("Sensor Response" if sensor_resp_time else "Source", rt_label)

            if source:
                pct_bar = max(0.0, min(1.0, 1.0 - dist / 50.0))
                bar_txt = f"Positioning Length: {dist:.1f} cm — {'✓ IN POSITION' if dist <= 10 else 'Move assembly closer to sensor'}"
                st.progress(pct_bar, text=bar_txt)

                if dist <= 10:
                    st.success(f"✅ Positioning confirmed — {dist:.1f} cm ≤ 10 cm target")
                else:
                    st.info(f"📏 Current: {dist:.1f} cm → needs to be ≤ 10 cm")

            # Fallback simulate
            with st.expander("⚙ No NodeMCU? Use simulation"):
                st.caption("Simulate sensor reading for testing.")
                if st.button("Simulate Reading (−12 cm)", use_container_width=True):
                    st.session_state.sensor_dist   = max(4.0, dist - 12.0)
                    st.session_state.sensor_source = "simulated"
                    st.rerun()

            st.divider()

            # ── Final pass gate ─────────────────────────
            pos_ok = source and dist <= 10
            if torque_ok and pos_ok:
                st.success(f"✅  {s['success']}")
                if st.button("✅  COMPLETE ASSEMBLY", use_container_width=True, type="primary"):
                    st.session_state.done.append(st.session_state.stage)
                    st.session_state.screen = "complete"
                    st.rerun()
            else:
                pending = []
                if not torque_ok:
                    pending.append("Apply correct torque (13–17 Nm)")
                if not pos_ok:
                    pending.append("Read sensor ≤ 10 cm positioning")
                st.warning("Pending: " + " · ".join(pending))

    # ── RIGHT: Parts + Torque + Positioning + Video + Instructions
    with R:
        # Torque & Positioning specs
        st.markdown('<div class="sec-title">Specs</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="display:flex;gap:6px;margin-bottom:8px;">'
            f'<div style="flex:1;background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.3);'
            f'border-radius:6px;padding:8px 10px;">'
            f'<div style="font-size:9px;color:#64748b;letter-spacing:2px;">TORQUE</div>'
            f'<div style="font-size:16px;font-weight:700;color:#f59e0b;font-family:monospace;">{s["torque"]}</div>'
            f'<div style="font-size:9px;color:#64748b;margin-top:2px;">{s["torque_note"]}</div></div>'
            f'<div style="flex:1;background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.3);'
            f'border-radius:6px;padding:8px 10px;">'
            f'<div style="font-size:9px;color:#64748b;letter-spacing:2px;">POSITION</div>'
            f'<div style="font-size:16px;font-weight:700;color:#10b981;font-family:monospace;">{s["pos_length"]}</div>'
            f'<div style="font-size:9px;color:#64748b;margin-top:2px;">{s["pos_note"]}</div></div>'
            f'</div>',
            unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Parts Required</div>', unsafe_allow_html=True)
        for num, name, note, hi in s["parts"]:
            cls = "pp-hi" if hi else "pp-nm"
            st.markdown(
                f'<div class="pp {cls}">'
                f'<div class="pp-num">{num}</div>'
                f'<div><div class="pp-name">{name}</div>'
                f'<div class="pp-note">{note}</div></div></div>',
                unsafe_allow_html=True)

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Assembly Video</div>', unsafe_allow_html=True)
        if s["video"]:
            vpath = os.path.join(BASE_DIR, s["video"])
            if os.path.exists(vpath):
                st.video(vpath, loop=True, autoplay=True, muted=True)
            else:
                st.caption(f"⚠ {s['video']} not found")
        else:
            st.caption("No video for this stage")

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Step Instructions</div>', unsafe_allow_html=True)
        for i, instr in enumerate(s["instructions"]):
            st.markdown(
                f'<div class="is"><span style="color:#f59e0b;margin-right:6px;">{i+1}.</span>{instr}</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SCREEN: COMPLETE
# ══════════════════════════════════════════════════════════
elif st.session_state.screen == "complete":
    elapsed = int(time.time() - (st.session_state.start_time or time.time()))
    mins, secs = divmod(elapsed, 60)

    st.markdown(f"""
    <div style="text-align:center;padding:48px 0 24px;">
      <div style="font-size:76px;">✅</div>
      <div style="font-size:46px;font-weight:800;color:#10b981;letter-spacing:2px;margin-top:8px;">
        ASSEMBLY COMPLETE
      </div>
      <div style="color:#64748b;margin-top:8px;font-size:14px;">
        All 5 stages verified · Quality check passed
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stages Done",    "5 / 5")
    c2.metric("Time Taken",     f"{mins}:{secs:02d}")
    c3.metric("Final Assembly", "A4")
    c4.metric("QC Status",      "PASS ✓")

    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
    _, mc, _ = st.columns([1,2,1])
    with mc:
        if st.button("＋  NEW ASSEMBLY", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
