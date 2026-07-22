# SMPL Interactive Viewer + Sensor Annotation + UV Map Selection
#
# Required local assets are not included in this repository.
# Users must download the official SMPL model and UV files separately.

import inspect

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import threading
import webbrowser
from functools import lru_cache
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from smplx import SMPL


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "body_models" / "smpl"
UV_DIR = BASE_DIR / "body_models" / "smpl_uv_20200910"

SMPL_GENDERS = {"MALE", "FEMALE", "NEUTRAL"}
smpl_models = {}


def find_uv_obj():
    """Find the official SMPL UV OBJ inside UV_DIR."""
    preferred_names = (
        "smpl_uv.obj",
        "SMPL_uv.obj",
        "smpl_uv_20200910.obj",
    )

    for name in preferred_names:
        candidate = UV_DIR / name
        if candidate.is_file():
            return candidate

    if UV_DIR.is_dir():
        obj_files = sorted(UV_DIR.rglob("*.obj"))
        if len(obj_files) == 1:
            return obj_files[0]
        if obj_files:
            for candidate in obj_files:
                if "smpl" in candidate.name.lower() and "uv" in candidate.name.lower():
                    return candidate

    return UV_DIR / "smpl_uv.obj"


def validate_required_files():
    """Check that the user has supplied the official licensed assets."""
    missing_models = []

    for gender in sorted(SMPL_GENDERS):
        model_file = MODEL_DIR / f"SMPL_{gender}.pkl"
        if not model_file.is_file():
            missing_models.append(model_file.relative_to(BASE_DIR))

    if missing_models:
        formatted = "\n".join(f"  - {path}" for path in missing_models)
        raise FileNotFoundError(
            "Missing official SMPL model files:\n"
            f"{formatted}\n\n"
            "Download them from the official SMPL website and place them in:\n"
            f"  {MODEL_DIR}"
        )

    uv_obj_path = find_uv_obj()
    if not uv_obj_path.is_file():
        raise FileNotFoundError(
            "Could not find the official SMPL UV OBJ file.\n\n"
            "Download and extract smpl_uv_20200910 under:\n"
            f"  {UV_DIR}\n\n"
            "The program searches this folder recursively for an .obj file."
        )


def get_model(gender):
    """Load and cache an SMPL model."""
    normalized_gender = str(gender).upper()
    if normalized_gender not in SMPL_GENDERS:
        normalized_gender = "NEUTRAL"

    if normalized_gender not in smpl_models:
        smpl_models[normalized_gender] = SMPL(
            model_path=str(MODEL_DIR),
            gender=normalized_gender,
            batch_size=1,
        )

    return smpl_models[normalized_gender]


def _resolve_obj_index(raw_index, item_count):
    """Convert a one-based or negative OBJ index to a zero-based index."""
    index = int(raw_index)
    if index > 0:
        return index - 1
    if index < 0:
        return item_count + index
    raise ValueError("OBJ indices cannot be zero.")


@lru_cache(maxsize=1)
def load_uv_data():
    """Parse the official SMPL UV OBJ into structures used by the viewer."""
    obj_path = find_uv_obj()

    uv_coords = []
    faces_v = []
    faces_vt = []
    vertex_to_uv = {}

    with obj_path.open("r", encoding="utf-8", errors="ignore") as obj_file:
        for line_number, raw_line in enumerate(obj_file, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("vt "):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(
                        f"Invalid texture coordinate on OBJ line {line_number}."
                    )
                uv_coords.append([float(parts[1]), float(parts[2])])
                continue

            if not line.startswith("f "):
                continue

            tokens = line.split()[1:]
            if len(tokens) < 3:
                continue

            for i in range(1, len(tokens) - 1):
                triangle_tokens = (tokens[0], tokens[i], tokens[i + 1])
                triangle_v = []
                triangle_vt = []

                for token in triangle_tokens:
                    fields = token.split("/")
                    if len(fields) < 2 or not fields[1]:
                        raise ValueError(
                            "The UV OBJ contains a face without a texture index "
                            f"on line {line_number}: {token}"
                        )

                    vertex_index = _resolve_obj_index(fields[0], 6890)
                    uv_index = _resolve_obj_index(fields[1], len(uv_coords))

                    if not 0 <= uv_index < len(uv_coords):
                        raise ValueError(
                            f"Invalid UV index on OBJ line {line_number}: {token}"
                        )

                    triangle_v.append(vertex_index)
                    triangle_vt.append(uv_index)
                    vertex_to_uv.setdefault(vertex_index, uv_coords[uv_index])

                faces_v.append(triangle_v)
                faces_vt.append(triangle_vt)

    model = get_model("NEUTRAL")
    vertex_count = int(model.v_template.shape[0])
    vertex_uv = [None] * vertex_count

    for vertex_index, uv in vertex_to_uv.items():
        if 0 <= vertex_index < vertex_count:
            vertex_uv[vertex_index] = uv

    missing_vertices = [
        index for index, coordinate in enumerate(vertex_uv)
        if coordinate is None
    ]
    if missing_vertices:
        raise ValueError(
            f"The supplied UV OBJ has no UV mapping for "
            f"{len(missing_vertices)} SMPL vertices."
        )

    if not uv_coords or not faces_vt:
        raise ValueError("No usable UV data was found in the supplied OBJ file.")

    return {
        "vertex_uv": vertex_uv,
        "uv_coords": uv_coords,
        "faces_v": faces_v,
        "faces_vt": faces_vt,
        "source_file": str(obj_path.relative_to(BASE_DIR)),
    }


@app.route("/get_mesh", methods=["POST"])
def get_mesh():
    data = request.get_json(silent=True) or {}

    gender = data.get("gender", "NEUTRAL")
    betas_values = data.get("betas", [0] * 10)
    pose_values = data.get("pose", [0] * 69)
    global_values = data.get("global_orient", [0] * 3)

    if len(betas_values) != 10:
        return jsonify({"error": "betas must contain 10 values."}), 400
    if len(pose_values) != 69:
        return jsonify({"error": "pose must contain 69 values."}), 400
    if len(global_values) != 3:
        return jsonify({"error": "global_orient must contain 3 values."}), 400

    try:
        model = get_model(gender)
        betas = torch.tensor([betas_values], dtype=torch.float32)
        pose = torch.tensor([pose_values], dtype=torch.float32)
        global_orient = torch.tensor([global_values], dtype=torch.float32)

        with torch.no_grad():
            output = model(
                betas=betas,
                body_pose=pose,
                global_orient=global_orient,
            )

        return jsonify({
            "vertices": output.vertices.detach().cpu().numpy()[0].tolist(),
            "faces": model.faces.tolist(),
        })
    except (TypeError, ValueError, RuntimeError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 500


@app.route("/get_uv", methods=["GET"])
def get_uv():
    try:
        return jsonify(load_uv_data())
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        return jsonify({"error": str(error)}), 500


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SMPL Sensor Annotator</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
:root{
  --bg:#0d0f14;--panel:#13161d;--border:#1e2330;
  --accent:#5b8cff;--accent2:#a67cff;--muted:#5a6080;
  --text:#d4d8e8;--pos:#5bffb0;--neg:#ff6b8a;--warn:#ffd06b;
  --sensor:#ff4d6d;--sensor-sel:#ffb703;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:'SF Mono','Fira Code',monospace;display:flex;height:100vh;overflow:hidden;}
#left{width:260px;min-width:260px;background:var(--panel);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;}
.ph{padding:13px 15px 9px;border-bottom:1px solid var(--border);flex-shrink:0;}
.ph h1{font-size:11px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);}
.ph p{font-size:9px;color:var(--muted);margin-top:2px;}
.tabs{display:flex;border-bottom:1px solid var(--border);flex-shrink:0;}
.tab{flex:1;padding:7px 0;font-family:inherit;font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;background:transparent;border:none;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;transition:all .15s;}
.tab.active{color:var(--accent);border-bottom-color:var(--accent);}
.tab-body{flex:1;overflow-y:auto;display:none;flex-direction:column;}
.tab-body.active{display:flex;}
.tab-body::-webkit-scrollbar{width:3px;}
.tab-body::-webkit-scrollbar-thumb{background:var(--border);}
.gr{display:flex;gap:5px;padding:9px 15px;border-bottom:1px solid var(--border);}
.gb{flex:1;padding:5px 0;border:1px solid var(--border);background:transparent;color:var(--muted);font-family:inherit;font-size:9px;letter-spacing:.06em;text-transform:uppercase;border-radius:3px;cursor:pointer;transition:all .15s;}
.gb.active{background:var(--accent);border-color:var(--accent);color:#fff;}
.sec{padding:9px 15px;border-bottom:1px solid var(--border);}
.st{font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:7px;display:flex;justify-content:space-between;align-items:center;}
.st button{font-family:inherit;font-size:8px;background:transparent;border:1px solid var(--border);color:var(--muted);padding:2px 5px;border-radius:3px;cursor:pointer;}
.st button:hover{border-color:var(--accent);color:var(--accent);}
.sr{margin-bottom:7px;}
.sl{display:flex;justify-content:space-between;margin-bottom:2px;}
.sl span:first-child{font-size:10px;}
.sv{font-size:10px;font-weight:700;min-width:30px;text-align:right;}
input[type=range]{-webkit-appearance:none;width:100%;height:3px;border-radius:2px;background:var(--border);outline:none;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:10px;height:10px;border-radius:50%;background:var(--accent);cursor:pointer;}
.js{display:flex;gap:5px;align-items:center;margin-bottom:7px;}
.js label{font-size:9px;color:var(--muted);white-space:nowrap;}
.js select{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);font-family:inherit;font-size:9px;padding:3px 5px;border-radius:3px;outline:none;}
#resetAll{margin:8px 15px;padding:6px;background:transparent;border:1px solid var(--border);color:var(--muted);font-family:inherit;font-size:9px;letter-spacing:.1em;text-transform:uppercase;border-radius:3px;cursor:pointer;}
#resetAll:hover{border-color:var(--neg);color:var(--neg);}
.sensor-list{flex:1;overflow-y:auto;padding:3px 0;}
.sensor-list::-webkit-scrollbar{width:3px;}
.sensor-list::-webkit-scrollbar-thumb{background:var(--border);}
.sensor-item{display:flex;align-items:center;gap:6px;padding:4px 15px;cursor:pointer;border-left:2px solid transparent;}
.sensor-item:hover{background:rgba(255,255,255,.03);}
.sensor-item.selected{background:rgba(255,183,3,.07);border-left-color:var(--sensor-sel);}
.sensor-item.assigned .sdot{box-shadow:0 0 0 2px var(--sensor-sel);}
.sdot{width:7px;height:7px;border-radius:50%;background:var(--sensor);flex-shrink:0;}
.sensor-item.selected .sdot{background:var(--sensor-sel);}
.sinfo{flex:1;min-width:0;}
.sidx{font-size:10px;color:var(--text);}
.scoord{font-size:8px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sgrid{font-size:8px;color:var(--sensor-sel);margin-top:1px;}
.sdel{background:transparent;border:none;color:var(--muted);cursor:pointer;font-size:10px;padding:0 2px;flex-shrink:0;}
.sdel:hover{color:var(--neg);}
.sensor-empty{padding:18px 15px;font-size:10px;color:var(--muted);text-align:center;line-height:1.8;}
.sensor-actions{padding:7px 15px;border-top:1px solid var(--border);display:flex;gap:5px;flex-shrink:0;}
.act-btn{flex:1;padding:6px 0;border:1px solid var(--border);background:transparent;color:var(--muted);font-family:inherit;font-size:8px;letter-spacing:.08em;text-transform:uppercase;border-radius:3px;cursor:pointer;transition:all .15s;}
.act-btn:hover{border-color:var(--accent);color:var(--accent);}
.act-btn.export{border-color:var(--accent2);color:var(--accent2);}
.act-btn.export:hover{background:var(--accent2);color:#fff;}
.act-btn.danger:hover{border-color:var(--neg);color:var(--neg);}
#status{padding:7px 15px;font-size:9px;color:var(--muted);display:flex;align-items:center;gap:5px;flex-shrink:0;border-top:1px solid var(--border);}
.dot{width:5px;height:5px;border-radius:50%;background:var(--muted);flex-shrink:0;transition:background .2s;}
.dot.loading{background:var(--warn);animation:pulse .8s infinite;}
.dot.ok{background:var(--pos);}.dot.err{background:var(--neg);}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* 3D VIEWPORT */
#viewport{flex:1;position:relative;cursor:grab;}
#viewport.dragging{cursor:grabbing;}
#viewport canvas{display:block;width:100%;height:100%;}
.vhint{position:absolute;bottom:10px;right:12px;font-size:9px;color:var(--muted);pointer-events:none;text-align:right;line-height:1.7;}

/* UV PANEL */
#uvpanel{width:460px;min-width:460px;background:var(--panel);border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;}
.uph{padding:13px 15px 9px;border-bottom:1px solid var(--border);flex-shrink:0;display:flex;justify-content:space-between;align-items:center;}
.uph h2{font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);}
.uph-right{display:flex;align-items:center;gap:10px;}
.uph span{font-size:9px;color:var(--muted);}
.uv-hint{padding:6px 15px;font-size:9px;color:var(--muted);border-bottom:1px solid var(--border);flex-shrink:0;}
.uv-hint b{color:var(--sensor-sel);}
#uv-wrap{flex:1;overflow:auto;display:flex;align-items:flex-start;justify-content:flex-start;padding:10px;position:relative;}
#uv-wrap::-webkit-scrollbar{width:5px;height:5px;}
#uv-wrap::-webkit-scrollbar-thumb{background:var(--border);}
#uv-canvas{cursor:crosshair;display:block;image-rendering:pixelated;}
.uv-footer{padding:7px 15px;border-top:1px solid var(--border);font-size:9px;color:var(--muted);flex-shrink:0;display:flex;justify-content:space-between;align-items:center;}
.zoom-controls{display:flex;gap:5px;align-items:center;}
.zoom-btn{background:transparent;border:1px solid var(--border);color:var(--muted);font-family:inherit;font-size:10px;width:22px;height:22px;border-radius:3px;cursor:pointer;transition:all .15s;}
.zoom-btn:hover{border-color:var(--accent);color:var(--accent);}

/* GRID PANEL */
#gridpanel{width:380px;min-width:380px;background:var(--panel);border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;}
.rph{padding:13px 15px 9px;border-bottom:1px solid var(--border);flex-shrink:0;display:flex;justify-content:space-between;align-items:center;}
.rph h2{font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);}
.rph span{font-size:9px;color:var(--muted);}
.grid-hint{padding:6px 15px;font-size:9px;color:var(--muted);border-bottom:1px solid var(--border);flex-shrink:0;line-height:1.6;}
.grid-hint b{color:var(--sensor-sel);}
#grid-area{flex:1;overflow:auto;padding:8px 10px 10px;}
#grid-area::-webkit-scrollbar{width:4px;height:4px;}
#grid-area::-webkit-scrollbar-thumb{background:var(--border);}
.grid-outer{display:inline-flex;flex-direction:column;}
.grid-row-wrap{display:flex;align-items:center;gap:3px;}
.row-label{width:16px;font-size:7px;color:var(--muted);text-align:right;flex-shrink:0;}
.col-labels{display:flex;gap:2px;margin-left:19px;margin-bottom:2px;}
.col-label{width:16px;font-size:7px;color:var(--muted);text-align:center;flex-shrink:0;}
#grid-table{display:flex;flex-direction:column;gap:2px;}
.grid-row{display:flex;gap:2px;}
.cell{width:16px;height:16px;border-radius:2px;background:var(--border);border:1px solid rgba(255,255,255,.04);cursor:pointer;transition:all .1s;flex-shrink:0;}
.cell:hover{background:#2a3050;border-color:var(--accent);}
.cell.assigned{background:var(--sensor);border-color:rgba(255,77,109,.6);}
.cell.assigned:hover{background:#ff7090;}
.cell.highlight{background:rgba(255,183,3,.3);border-color:var(--sensor-sel);}
.cell.assigned.highlight{background:var(--sensor-sel);border-color:#ffcf40;}
.grid-footer{padding:7px 15px;border-top:1px solid var(--border);font-size:9px;color:var(--muted);flex-shrink:0;}
</style>
</head>
<body>

<!-- LEFT: controls + sensor list -->
<div id="left">
  <div class="ph"><h1>SMPL Annotator</h1><p>Shape, pose &amp; sensor placement</p></div>
  <div class="tabs">
    <button class="tab active" onclick="switchTab('body',this)">Body</button>
    <button class="tab" onclick="switchTab('sensor',this)">Sensors</button>
  </div>
  <div class="tab-body active" id="tab-body">
    <div class="gr">
      <button class="gb" data-g="MALE" onclick="setGender(this)">Male</button>
      <button class="gb active" data-g="NEUTRAL" onclick="setGender(this)">Neutral</button>
      <button class="gb" data-g="FEMALE" onclick="setGender(this)">Female</button>
    </div>
    <div class="sec"><div class="st">Shape Betas<button onclick="resetBetas()">Reset</button></div><div id="bs"></div></div>
    <div class="sec">
      <div class="st">Joint Pose<button onclick="resetPose()">Reset</button></div>
      <div class="js"><label>Joint</label><select id="jsel" onchange="onJointChange()"></select></div>
      <div id="ps"></div>
    </div>
    <div class="sec"><div class="st">Global Orient<button onclick="resetOrient()">Reset</button></div><div id="os"></div></div>
    <button id="resetAll" onclick="resetAll()">&#8635; Reset All</button>
  </div>
  <div class="tab-body" id="tab-sensor">
    <div class="sensor-list" id="slist">
      <div class="sensor-empty" id="sempty">No sensors yet.<br>Click on the UV map<br>to place sensors.</div>
    </div>
    <div class="sensor-actions">
      <button class="act-btn danger" onclick="clearSensors()">Clear All</button>
      <button class="act-btn export" onclick="exportJSON()">Export JSON</button>
    </div>
  </div>
  <div id="status"><div class="dot ok" id="dot"></div><span id="stxt">Ready</span></div>
</div>

<!-- CENTER: 3D preview -->
<div id="viewport">
  <canvas id="cv3d"></canvas>
  <div class="vhint">Left-drag rotate · Scroll zoom<br>Middle / Shift+drag pan</div>
</div>

<!-- RIGHT-1: UV map -->
<div id="uvpanel">
  <div class="uph">
    <h2>UV Map &mdash; Click to Place Sensor</h2>
    <div class="uph-right">
      <span id="uv-sensor-hint">Click UV to add sensor</span>
    </div>
  </div>
  <div class="uv-hint"><b>Click</b> on the UV map to place a sensor. <b>Click existing dot</b> to remove it. <b>Scroll</b> to zoom · <b>Right/middle drag</b> to pan.</div>
  <div id="uv-wrap"><canvas id="uv-canvas"></canvas></div>
  <div class="uv-footer">
    <span id="uv-coord">Hover over UV map</span>
    <div class="zoom-controls">
      <button class="zoom-btn" onclick="uvZoom(-1)">&#8722;</button>
      <span id="zoom-label" style="font-size:9px;color:var(--muted);min-width:36px;text-align:center;">2.0x</span>
      <button class="zoom-btn" onclick="uvZoom(1)">&#43;</button>
    </div>
  </div>
</div>

<!-- RIGHT-2: 32x16 grid -->
<div id="gridpanel">
  <div class="rph"><h2>32 &times; 16 Grid</h2><span id="grid-stat">0 / 512 assigned</span></div>
  <div class="grid-hint"><b>Select a sensor</b> in the list, then click a cell to assign it.</div>
  <div id="grid-area">
    <div class="grid-outer">
      <div class="col-labels" id="col-labels"></div>
      <div id="grid-table"></div>
    </div>
  </div>
  <div class="grid-footer" id="gf-stat">Select a sensor to assign to grid.</div>
</div>

<script>
// ── UV DATA (loaded from the user's local official OBJ) ──────────────────────
var VERT_UV=[];
var UV_COORDS=[];
var FACES_V=[];
var FACES_VT=[];
var uvDataReady=false;

var JOINTS=['L Hip','R Hip','Spine','L Knee','R Knee','Spine2','L Ankle','R Ankle',
  'Spine3','L Foot','R Foot','Neck','L Collar','R Collar','Head','L Shoulder',
  'R Shoulder','L Elbow','R Elbow','L Wrist','R Wrist','L Hand','R Hand'];

var S={gender:'NEUTRAL',betas:new Array(10).fill(0),pose:new Array(69).fill(0),go:new Array(3).fill(0)};
var selJ=0,bodyTimer=null;

var sensors=[],sensorIdCounter=1,selectedSensorIdx=-1;
var grid=[];
for(var r=0;r<ROWS;r++) grid.push(new Array(COLS).fill(0));
var currentVertices=[];

// UV canvas state
var uvScale=2.0;
var UV_SIZE=512;
var uvPanX=0, uvPanY=0;
var uvDragging=false, uvDragLast={x:0,y:0};

// ── Tabs ──────────────────────────────────────────────────────────────────────
function switchTab(name,btn){
  document.querySelectorAll('.tab').forEach(function(t){t.classList.remove('active');});
  document.querySelectorAll('.tab-body').forEach(function(t){t.classList.remove('active');});
  btn.classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
}

// ── Sliders ───────────────────────────────────────────────────────────────────
function mkSlider(id,label,min,max,step,val,unit,cb){
  var d=document.createElement('div');d.className='sr';
  d.innerHTML='<div class="sl"><span>'+label+'</span><span class="sv" id="v_'+id+'">'+val+unit+'</span></div>'
    +'<input type="range" id="'+id+'" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val+'">';
  d.querySelector('input').addEventListener('input',function(e){
    var v=parseFloat(e.target.value),ve=document.getElementById('v_'+id);
    ve.textContent=(Number.isInteger(v)?v:v.toFixed(2))+unit;
    ve.style.color=v>0?'var(--pos)':v<0?'var(--neg)':'var(--muted)';
    cb(v);
  });
  return d;
}
var bc=document.getElementById('bs');
for(var _i=0;_i<10;_i++)(function(i){bc.appendChild(mkSlider('b'+i,'Shape '+(i+1),-3,3,0.05,0,'',function(v){S.betas[i]=v;sched();}));}(_i));
var jsel=document.getElementById('jsel');
JOINTS.forEach(function(n,i){var o=document.createElement('option');o.value=i;o.textContent=i+': '+n;jsel.appendChild(o);});
var pc=document.getElementById('ps');
['X Axis','Y Axis','Z Axis'].forEach(function(l,ax){
  pc.appendChild(mkSlider('p'+ax,l,-180,180,1,0,'deg',function(v){S.pose[selJ*3+ax]=v*Math.PI/180;sched();}));
});
var oc=document.getElementById('os');
['X Axis','Y Axis','Z Axis'].forEach(function(l,ax){
  oc.appendChild(mkSlider('o'+ax,l,-180,180,1,0,'deg',function(v){S.go[ax]=v*Math.PI/180;sched();}));
});
function onJointChange(){
  selJ=parseInt(jsel.value);
  [0,1,2].forEach(function(ax){
    var deg=Math.round(S.pose[selJ*3+ax]*180/Math.PI);
    var el=document.getElementById('p'+ax),ve=document.getElementById('v_p'+ax);
    if(el){el.value=deg;ve.textContent=deg+'deg';ve.style.color=deg>0?'var(--pos)':deg<0?'var(--neg)':'var(--muted)';}
  });
}
function setVal(id,v,unit){unit=unit||'';var el=document.getElementById(id),ve=document.getElementById('v_'+id);if(el){el.value=v;ve.textContent=v+unit;ve.style.color='var(--muted)';}}
function resetBetas(){S.betas.fill(0);for(var i=0;i<10;i++)setVal('b'+i,0);sched();}
function resetPose(){S.pose.fill(0);[0,1,2].forEach(function(ax){setVal('p'+ax,0,'deg');});sched();}
function resetOrient(){S.go.fill(0);[0,1,2].forEach(function(ax){setVal('o'+ax,0,'deg');});sched();}
function resetAll(){resetBetas();resetPose();resetOrient();}
function setGender(btn){document.querySelectorAll('.gb').forEach(function(b){b.classList.remove('active');});btn.classList.add('active');S.gender=btn.dataset.g;sched();}
function setStatus(t,msg){document.getElementById('dot').className='dot '+t;document.getElementById('stxt').textContent=msg;}
function sched(){clearTimeout(bodyTimer);bodyTimer=setTimeout(fetchMesh,120);}

// ── Load UV data ───────────────────────────────────────────────────────────────
async function fetchUVData(){
  setStatus('loading','Loading UV data...');
  try{
    var response=await fetch('/get_uv');
    var data=await response.json();
    if(!response.ok) throw new Error(data.error||('HTTP '+response.status));

    VERT_UV=data.vertex_uv;
    UV_COORDS=data.uv_coords;
    FACES_V=data.faces_v;
    FACES_VT=data.faces_vt;
    uvDataReady=true;
    uvSpatialIndex=null;

    buildUVSpatialIndex();
    drawUV();

    var hint=document.getElementById('uv-sensor-hint');
    if(hint) hint.textContent='Click UV to add sensor';
    setStatus('ok','UV data loaded');
  }catch(error){
    uvDataReady=false;
    var hint=document.getElementById('uv-sensor-hint');
    if(hint) hint.textContent='Official SMPL UV OBJ not found';
    setStatus('err','UV error: '+error.message);
    console.error(error);
    throw error;
  }
}

// ── Fetch mesh ─────────────────────────────────────────────────────────────────
async function fetchMesh(){
  setStatus('loading','Computing...');
  try{
    var r=await fetch('/get_mesh',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({gender:S.gender,betas:S.betas,pose:S.pose,global_orient:S.go})});
    if(!r.ok) throw new Error('HTTP '+r.status);
    var d=await r.json();
    currentVertices=d.vertices;
    updateMesh3D(d.vertices,d.faces);
    refreshSensorPositions();
    setStatus('ok','Updated');
  }catch(e){setStatus('err','Error: '+e.message);}
}

// ── Sensors ────────────────────────────────────────────────────────────────────
function addSensorByVertex(vertexIndex){
  if(!uvDataReady||!VERT_UV.length){
    setStatus('err','UV data has not been loaded');
    return;
  }
  if(vertexIndex<0||vertexIndex>=VERT_UV.length||!VERT_UV[vertexIndex]){
    setStatus('err','No UV coordinate for vertex '+vertexIndex);
    return;
  }

  var existing=sensors.findIndex(function(s){return s.vertexIndex===vertexIndex;});
  if(existing!==-1){deleteSensor(existing);return;}

  var v=currentVertices.length?currentVertices[vertexIndex]:[0,0,0];
  var uv=VERT_UV[vertexIndex];
  sensors.push({
    id:sensorIdCounter++,vertexIndex:vertexIndex,
    x:v[0],y:v[1],z:v[2],u:uv[0],v_:uv[1],row:-1,col:-1
  });

  selectedSensorIdx=sensors.length-1;
  rebuild3DMarkers();
  renderSensorList();
  drawUV();
  setStatus('ok','Sensor #'+sensors[selectedSensorIdx].id+' placed');
}

function deleteSensor(idx){
  var s=sensors[idx];
  if(s.row>=0&&s.col>=0) grid[s.row][s.col]=0;
  sensors.splice(idx,1);
  if(selectedSensorIdx===idx) selectedSensorIdx=-1;
  else if(selectedSensorIdx>idx) selectedSensorIdx--;
  rebuild3DMarkers();renderSensorList();drawUV();renderGrid();
}

function selectSensor(idx){
  selectedSensorIdx=idx;
  renderSensorList();drawUV();renderGrid();
}

function assignCell(row,col){
  if(selectedSensorIdx<0||selectedSensorIdx>=sensors.length) return;
  var s=sensors[selectedSensorIdx];
  if(grid[row][col]!==0&&grid[row][col]!==s.id){
    var occ=sensors.findIndex(function(x){return x.id===grid[row][col];});
    if(occ>=0){sensors[occ].row=-1;sensors[occ].col=-1;}
  }
  if(s.row===row&&s.col===col){grid[row][col]=0;s.row=-1;s.col=-1;}
  else{if(s.row>=0&&s.col>=0)grid[s.row][s.col]=0;grid[row][col]=s.id;s.row=row;s.col=col;}
  renderSensorList();renderGrid();updateGridStat();
}

function clearSensors(){
  sensors=[];selectedSensorIdx=-1;
  for(var r=0;r<ROWS;r++)for(var c=0;c<COLS;c++)grid[r][c]=0;
  rebuild3DMarkers();renderSensorList();drawUV();renderGrid();updateGridStat();
}

function updateGridStat(){
  var n=sensors.filter(function(s){return s.row>=0;}).length;
  document.getElementById('grid-stat').textContent=n+' / 512 assigned';
}

function exportJSON(){
  if(!sensors.length){
    setStatus('err','No sensors to export');
    return;
  }

  var exportedSensors=sensors.map(function(s){
    return {
      sensor_id:s.id,
      vertex_index:s.vertexIndex,

      position:{
        x:s.x,
        y:s.y,
        z:s.z
      },

      uv:{
        u:s.u,
        v:s.v_
      },

      grid:{
        row:s.row>=0?s.row+1:null,
        column:s.col>=0?s.col+1:null
      }
    };
  });

  var gridData=[];

  for(var r=0;r<ROWS;r++){
    var rowData=[];

    for(var c=0;c<COLS;c++){
      var sensorId=grid[r][c];

      if(!sensorId){
        rowData.push(null);
        continue;
      }

      var sensor=sensors.find(function(item){
        return item.id===sensorId;
      });

      rowData.push(
        sensor
          ? {
              sensor_id:sensor.id,
              vertex_index:sensor.vertexIndex
            }
          : null
      );
    }

    gridData.push(rowData);
  }

  var output={
    format:"SMPL Sensor Annotation",
    version:"1.0",

    model:{
      type:"SMPL",
      gender:S.gender,
      betas:S.betas.slice(),
      body_pose:S.pose.slice(),
      global_orient:S.go.slice(),
      coordinate_unit:"metres"
    },

    annotation:{
      sensor_count:sensors.length,
      assigned_sensor_count:sensors.filter(function(s){
        return s.row>=0&&s.col>=0;
      }).length,

      grid_size:{
        rows:ROWS,
        columns:COLS
      },

      sensors:exportedSensors,
      grid_32x16:gridData
    }
  };

  var jsonText=JSON.stringify(output,null,2);
  var blob=new Blob(
    [jsonText],
    {type:"application/json;charset=utf-8"}
  );

  var url=URL.createObjectURL(blob);
  var link=document.createElement('a');

  var timestamp=new Date()
    .toISOString()
    .replace(/[:.]/g,'-');

  link.href=url;
  link.download='smpl_sensor_annotation_'+timestamp+'.json';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);

  setStatus(
    'ok',
    'Exported '+sensors.length+' sensors'
  );
}

// ── Render sensor list ──────────────────────────────────────────────────────
function renderSensorList(){
  document.getElementById('sempty').style.display=sensors.length?'none':'';
  var list=document.getElementById('slist');
  list.querySelectorAll('.sensor-item').forEach(function(el){el.remove();});
  sensors.forEach(function(s,i){
    var item=document.createElement('div');
    item.className='sensor-item'+(i===selectedSensorIdx?' selected':'')+(s.row>=0?' assigned':'');
    item.innerHTML='<div class="sdot"></div>'
      +'<div class="sinfo"><div class="sidx">Sensor #'+s.id+' &middot; vtx '+s.vertexIndex+'</div>'
      +'<div class="scoord">'+s.x.toFixed(3)+', '+s.y.toFixed(3)+', '+s.z.toFixed(3)+'</div>'
      +(s.row>=0?'<div class="sgrid">Grid R'+(s.row+1)+' C'+(s.col+1)+'</div>':'')
      +'</div><button class="sdel" title="Delete">&#10005;</button>';
    item.addEventListener('click',function(e){
      if(e.target.classList.contains('sdel')) deleteSensor(i);
      else selectSensor(i);
    });
    list.appendChild(item);
  });
}

// ── UV Canvas ──────────────────────────────────────────────────────────────────
var uvCanvas,uvCtx;
// UV spatial index for fast nearest-vertex lookup

// Build a spatial index: for each UV cell in a grid, list vertex indices
// This lets us find nearest vertex to a click in O(1) average
var UV_GRID_N=64; // 64x64 buckets over [0,1]x[0,1]
var uvSpatialIndex=null;

function buildUVSpatialIndex(){
  uvSpatialIndex=[];
  if(!uvDataReady||!VERT_UV.length)return;
  for(var i=0;i<UV_GRID_N*UV_GRID_N;i++) uvSpatialIndex.push([]);
  for(var vi=0;vi<VERT_UV.length;vi++){
    var u=VERT_UV[vi][0], v=VERT_UV[vi][1];
    var bx=Math.min(UV_GRID_N-1,Math.floor(u*UV_GRID_N));
    var by=Math.min(UV_GRID_N-1,Math.floor(v*UV_GRID_N));
    uvSpatialIndex[by*UV_GRID_N+bx].push(vi);
  }
}

function findNearestVertex(u,v){
  // search 3x3 neighbourhood of buckets
  var bx=Math.min(UV_GRID_N-1,Math.max(0,Math.floor(u*UV_GRID_N)));
  var by=Math.min(UV_GRID_N-1,Math.max(0,Math.floor(v*UV_GRID_N)));
  var bestIdx=-1, bestDist=Infinity;
  for(var dy=-1;dy<=1;dy++){
    for(var dx=-1;dx<=1;dx++){
      var nx=bx+dx, ny=by+dy;
      if(nx<0||nx>=UV_GRID_N||ny<0||ny>=UV_GRID_N) continue;
      var bucket=uvSpatialIndex[ny*UV_GRID_N+nx];
      for(var k=0;k<bucket.length;k++){
        var vi=bucket[k];
        var du=VERT_UV[vi][0]-u, dv=VERT_UV[vi][1]-v;
        var d=du*du+dv*dv;
        if(d<bestDist){bestDist=d;bestIdx=vi;}
      }
    }
  }
  return bestIdx;
}

function uvZoom(dir){
  var minS=0.5, maxS=10;
  var factor = dir>0 ? 1.25 : 0.8;
  uvScale = Math.max(minS, Math.min(maxS, uvScale*factor));
  document.getElementById('zoom-label').textContent=uvScale.toFixed(1)+'x';
  clampUVPan();
  drawUV();
}

function clampUVPan(){
  var sz=UV_SIZE*uvScale;
  var wrap=document.getElementById('uv-wrap');
  var maxPX=Math.max(0, sz-wrap.clientWidth+20);
  var maxPY=Math.max(0, sz-wrap.clientHeight+20);
  uvPanX=Math.max(0,Math.min(uvPanX,maxPX));
  uvPanY=Math.max(0,Math.min(uvPanY,maxPY));
}

function resizeUVCanvas(){
  var wrap=document.getElementById('uv-wrap');
  uvCanvas.width=wrap.clientWidth;
  uvCanvas.height=wrap.clientHeight;
  uvCanvas.style.width=wrap.clientWidth+'px';
  uvCanvas.style.height=wrap.clientHeight+'px';
}

function initUVCanvas(){
  uvCanvas=document.getElementById('uv-canvas');
  uvCtx=uvCanvas.getContext('2d');
  resizeUVCanvas();
  buildUVSpatialIndex();
  drawUV();

  // Scroll to zoom (centered on cursor)
  uvCanvas.addEventListener('wheel',function(e){
    e.preventDefault();
    var rect=uvCanvas.getBoundingClientRect();
    var mx=e.clientX-rect.left, my=e.clientY-rect.top;
    // UV position under cursor before zoom
    var uBefore=(mx+uvPanX)/(UV_SIZE*uvScale);
    var vBefore=(my+uvPanY)/(UV_SIZE*uvScale);
    var factor=e.deltaY<0?1.15:0.87;
    var minS=0.5, maxS=12;
    uvScale=Math.max(minS,Math.min(maxS,uvScale*factor));
    // Adjust pan so cursor stays over same UV point
    uvPanX=uBefore*UV_SIZE*uvScale-mx;
    uvPanY=vBefore*UV_SIZE*uvScale-my;
    clampUVPan();
    document.getElementById('zoom-label').textContent=uvScale.toFixed(1)+'x';
    drawUV();
  },{passive:false});

  // Middle-click or right-click drag to pan
  uvCanvas.addEventListener('mousedown',function(e){
    if(e.button===1||e.button===2){e.preventDefault();uvDragging=true;uvDragLast={x:e.clientX,y:e.clientY};}
  });
  uvCanvas.addEventListener('mousemove',function(e){
    if(uvDragging){
      uvPanX-=e.clientX-uvDragLast.x;
      uvPanY-=e.clientY-uvDragLast.y;
      uvDragLast={x:e.clientX,y:e.clientY};
      clampUVPan();
      drawUV();
    }
    // UV coordinate display
    var rect=uvCanvas.getBoundingClientRect();
    var px=e.clientX-rect.left, py=e.clientY-rect.top;
    var u=((px+uvPanX)/(UV_SIZE*uvScale)).toFixed(3);
    var v=(1-(py+uvPanY)/(UV_SIZE*uvScale)).toFixed(3);
    document.getElementById('uv-coord').textContent='u='+u+' v='+v;
  });
  uvCanvas.addEventListener('mouseup',function(e){if(e.button===1||e.button===2)uvDragging=false;});
  uvCanvas.addEventListener('mouseleave',function(){uvDragging=false;});
  uvCanvas.addEventListener('contextmenu',function(e){e.preventDefault();});

  // Click to place sensor
  uvCanvas.addEventListener('click',function(e){
    if(uvDragging) return;
    var rect=uvCanvas.getBoundingClientRect();
    var px=e.clientX-rect.left, py=e.clientY-rect.top;
    var u=(px+uvPanX)/(UV_SIZE*uvScale);
    var v=1-(py+uvPanY)/(UV_SIZE*uvScale);
    if(u<0||u>1||v<0||v>1) return;
    var vi=findNearestVertex(u,v);
    if(vi>=0) addSensorByVertex(vi);
  });

  window.addEventListener('resize',function(){resizeUVCanvas();drawUV();});
}

function drawUV(){
  if(!uvCtx) return;
  var cw=uvCanvas.width, ch=uvCanvas.height;
  uvCtx.clearRect(0,0,cw,ch);

  // Background
  uvCtx.fillStyle='#0d0f14';
  uvCtx.fillRect(0,0,cw,ch);

  var sz=UV_SIZE*uvScale; // total UV map size in screen pixels

  // Apply pan via canvas transform
  uvCtx.save();
  uvCtx.translate(-uvPanX, -uvPanY);

  // Draw UV mesh wireframe
  uvCtx.strokeStyle='rgba(60,80,130,0.6)';
  uvCtx.lineWidth=Math.max(0.3, 0.4*uvScale);
  for(var fi=0;fi<FACES_VT.length;fi++){
    var fvt=FACES_VT[fi];
    uvCtx.beginPath();
    for(var k=0;k<3;k++){
      var uvi=fvt[k];
      var pu=UV_COORDS[uvi][0]*sz;
      var pv=(1-UV_COORDS[uvi][1])*sz;
      if(k===0) uvCtx.moveTo(pu,pv); else uvCtx.lineTo(pu,pv);
    }
    uvCtx.closePath();
    uvCtx.stroke();
  }

  // Draw sensor dots — fixed size regardless of zoom
  var dotR=Math.max(3, 4+uvScale*0.5);
  for(var i=0;i<sensors.length;i++){
    var s=sensors[i];
    var px=s.u*sz, py=(1-s.v_)*sz;
    var isSelected=(i===selectedSensorIdx);
    uvCtx.beginPath();
    uvCtx.arc(px,py,isSelected?dotR*1.4:dotR,0,Math.PI*2);
    uvCtx.fillStyle=isSelected?'#ffb703':'#ff4d6d';
    uvCtx.fill();
    if(isSelected){
      uvCtx.strokeStyle='rgba(255,183,3,0.6)';
      uvCtx.lineWidth=1.5;
      uvCtx.stroke();
    }
  }

  uvCtx.restore();
}

// ── 32x16 Grid ─────────────────────────────────────────────────────────────────
function buildGrid(){
  var cl=document.getElementById('col-labels');
  for(var c=0;c<COLS;c++){var l=document.createElement('div');l.className='col-label';l.textContent=c+1;cl.appendChild(l);}
  var gt=document.getElementById('grid-table');
  for(var r=0;r<ROWS;r++){
    var wrap=document.createElement('div');wrap.className='grid-row-wrap';
    var rl=document.createElement('div');rl.className='row-label';rl.textContent=r+1;wrap.appendChild(rl);
    var row=document.createElement('div');row.className='grid-row';
    for(var c2=0;c2<COLS;c2++){
      var cell=document.createElement('div');cell.className='cell';
      cell.dataset.r=r;cell.dataset.c=c2;
      cell.addEventListener('click',function(){assignCell(parseInt(this.dataset.r),parseInt(this.dataset.c));});
      cell.addEventListener('mouseenter',function(){
        var sid=grid[parseInt(this.dataset.r)][parseInt(this.dataset.c)];
        document.getElementById('gf-stat').textContent=
          'R'+(parseInt(this.dataset.r)+1)+' C'+(parseInt(this.dataset.c)+1)+(sid?': Sensor #'+sid+' assigned':': empty');
      });
      row.appendChild(cell);
    }
    wrap.appendChild(row);gt.appendChild(wrap);
  }
}

function getCellEl(r,c){return document.querySelector('.cell[data-r="'+r+'"][data-c="'+c+'"]');}

function renderGrid(){
  for(var r=0;r<ROWS;r++){
    for(var c=0;c<COLS;c++){
      var el=getCellEl(r,c);if(!el)continue;
      var sid=grid[r][c];
      var isSel=selectedSensorIdx>=0&&selectedSensorIdx<sensors.length
               &&sensors[selectedSensorIdx].row===r&&sensors[selectedSensorIdx].col===c;
      el.className='cell'+(sid?' assigned':'')+(isSel?' highlight':'');
    }
  }
  updateGridStat();
}

// ── Three.js (3D preview only, no clicking) ───────────────────────────────────
var renderer3,scene3,camera3,mesh3d,markers3;
var rx=0,ry=0,zoom3=2.8,panX=0,panY=0,modelYOffset=0;
var dragBtn=-1,lx=0,ly=0,moved=false;

function init3D(){
  var cv=document.getElementById('cv3d');
  renderer3=new THREE.WebGLRenderer({canvas:cv,antialias:true,alpha:true});
  renderer3.setPixelRatio(window.devicePixelRatio);
  scene3=new THREE.Scene();
  camera3=new THREE.PerspectiveCamera(40,1,0.01,100);
  camera3.position.set(0,0,zoom3);
  scene3.add(new THREE.AmbientLight(0xffffff,0.5));
  var dl=new THREE.DirectionalLight(0xffffff,1.0);dl.position.set(1,2,2);scene3.add(dl);
  scene3.add(new THREE.DirectionalLight(0x8899ff,0.3)).position.set(-1,-1,-1);
  mesh3d=new THREE.Mesh(new THREE.SphereGeometry(0.01),new THREE.MeshStandardMaterial());
  scene3.add(mesh3d);
  markers3=new THREE.Group();scene3.add(markers3);

  var vp=document.getElementById('viewport');
  vp.addEventListener('contextmenu',function(e){e.preventDefault();});
  vp.addEventListener('mousedown',function(e){
    if(e.button===1)e.preventDefault();
    dragBtn=e.button;lx=e.clientX;ly=e.clientY;moved=false;
    if(e.button===0&&!e.shiftKey) vp.classList.add('dragging');
  });
  vp.addEventListener('mousemove',function(e){
    if(dragBtn<0)return;
    var dx=e.clientX-lx,dy=e.clientY-ly;
    if(Math.abs(dx)>2||Math.abs(dy)>2)moved=true;
    if(dragBtn===1||(dragBtn===0&&e.shiftKey)){var sc=zoom3*.001;panX-=dx*sc;panY+=dy*sc;}
    else if(dragBtn===0&&!e.shiftKey){ry+=dx*.01;rx+=dy*.01;}
    lx=e.clientX;ly=e.clientY;
  });
  vp.addEventListener('mouseup',function(){dragBtn=-1;moved=false;vp.classList.remove('dragging');});
  vp.addEventListener('mouseleave',function(){dragBtn=-1;moved=false;vp.classList.remove('dragging');});
  vp.addEventListener('wheel',function(e){e.preventDefault();zoom3=Math.max(.3,Math.min(8,zoom3+e.deltaY*.004));},{passive:false});

  resize3D();window.addEventListener('resize',resize3D);
  (function loop(){
    requestAnimationFrame(loop);
    mesh3d.rotation.x=rx;mesh3d.rotation.y=ry;
    markers3.rotation.x=rx;markers3.rotation.y=ry;
    markers3.position.y=modelYOffset;
    camera3.position.set(panX,panY,zoom3);camera3.lookAt(panX,panY,0);
    renderer3.render(scene3,camera3);
  })();
}

function resize3D(){
  var vp=document.getElementById('viewport');
  renderer3.setSize(vp.clientWidth,vp.clientHeight,false);
  camera3.aspect=vp.clientWidth/vp.clientHeight;
  camera3.updateProjectionMatrix();
}

function updateMesh3D(verts,faces){
  scene3.remove(mesh3d);mesh3d.geometry.dispose();
  var geo=new THREE.BufferGeometry();
  var va=new Float32Array(verts.length*3);
  for(var i=0;i<verts.length;i++){va[i*3]=verts[i][0];va[i*3+1]=verts[i][1];va[i*3+2]=verts[i][2];}
  var ia=new Uint32Array(faces.length*3);
  for(var i=0;i<faces.length;i++){ia[i*3]=faces[i][0];ia[i*3+1]=faces[i][1];ia[i*3+2]=faces[i][2];}
  geo.setAttribute('position',new THREE.BufferAttribute(va,3));
  geo.setIndex(new THREE.BufferAttribute(ia,1));
  geo.computeVertexNormals();geo.computeBoundingBox();
  var cy=(geo.boundingBox.max.y+geo.boundingBox.min.y)/2;
  modelYOffset=-cy;
  mesh3d=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:0x4a7fff,roughness:.6,side:THREE.DoubleSide}));
  mesh3d.position.y=modelYOffset;mesh3d.rotation.x=rx;mesh3d.rotation.y=ry;
  scene3.add(mesh3d);
}

function rebuild3DMarkers(){
  while(markers3.children.length)markers3.remove(markers3.children[0]);
  var gN=new THREE.SphereGeometry(0.005,6,6);
  var mN=new THREE.MeshStandardMaterial({color:0xff4d6d,emissive:0xff2244,emissiveIntensity:.5});
  var gS=new THREE.SphereGeometry(0.007,6,6);
  var mS=new THREE.MeshStandardMaterial({color:0xffb703,emissive:0xffaa00,emissiveIntensity:.6});
  sensors.forEach(function(s,i){
    var isSel=(i===selectedSensorIdx);
    var m=new THREE.Mesh(isSel?gS:gN,isSel?mS:mN);
    m.position.set(s.x,s.y,s.z);
    markers3.add(m);
  });
}

function refreshSensorPositions(){
  if(!currentVertices.length)return;
  sensors.forEach(function(s){
    var v=currentVertices[s.vertexIndex];s.x=v[0];s.y=v[1];s.z=v[2];
  });
  rebuild3DMarkers();renderSensorList();renderGrid();
}

// ── Init ───────────────────────────────────────────────────────────────────────
async function initialiseApp(){
  buildGrid();
  initUVCanvas();
  init3D();

  try{
    await fetchUVData();
    await fetchMesh();
  }catch(error){
    console.error('Application initialisation failed:',error);
  }
}

initialiseApp();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return HTML


if __name__ == "__main__":
    try:
        validate_required_files()
    except FileNotFoundError as error:
        print("\nCannot start SMPL Sensor Annotator:\n")
        print(error)
        raise SystemExit(1)

    url = "http://localhost:5000"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"\n  SMPL Sensor Annotator -> {url}")
    print("  Close this terminal to quit.\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )