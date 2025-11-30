```markdown
# Smart-Canteen – Project Constitution  
**(Shape-Based Labeling Edition)**

> This constitution defines the long-term architectural principles, responsibilities, constraints, and directional rules for the **Smart-Canteen** AI-assisted vision checkout system.  
> All future specifications (`spec`), plans (`plan`), and tasks must comply with these foundational principles.  
> Any deviation must be explicitly justified and scheduled for remediation.

---

# 1. Project Overview

- **Project Name**: Smart-Canteen  
- **Goal**:  
  Deploy a robust, low-power, fully offline-capable cafeteria checkout system on x86 terminals.  
  The system detects **plate/bowl/tray shapes** via a USB camera and maps them to prices using cloud-hosted configuration.

- **Primary Users**:  
  - **Operator** – Manages checkout on the main screen.  
  - **Customer** – Observes mirrored display on the secondary screen.  
  - **Admin/Ops** – Manages configuration, updates, and maintenance.

- **Hardware Constraints**:  
  - Intel J1900 / i3 terminals, USB 1080p camera  
  - Dual displays  
  - Unstable or absent internet connectivity

---

# 2. Core Philosophy: Vision Decoupled from Business

### 2.1 AI = Eyes  
- The AI model outputs **only geometric/physical attributes**, not business meaning.  
- The output consists of:  
  `[{label, confidence, bbox}]`  
- Examples of valid labels:  
  - `plate_round_large`  
  - `bowl_round_small`  
  - `tray_rect_large`  

### 2.2 UI = Brain  
- Pricing, naming, discounts, and all business semantics live entirely in the **UI layer / business logic**.  
- AI labels are interpreted using `price_map.json`.  
- The UI has full authority to:
  - Assign names  
  - Set prices  
  - Filter disabled items  
  - Apply business rules  

### 2.3 Separation of Concerns  
- AI may **never** embed hardcoded prices or dish names.  
- UI may **never** perform visual inference or access the camera.  
- Cloud may **never** be required for real-time operation.

---

# 3. System Boundaries & Responsibilities

## 3.1 Local Python Service (Driver)

**Responsibilities**:
- Exclusive owner of USB camera  
- Runs YOLOv8n (OpenVINO/ONNX) for real-time inference  
- Serves MJPEG stream with bounding boxes  
- Exposes detection API for frontend  
- Performs OTA model updates  
- Provides local persistence and queueing for offline usage  

**Key Endpoints**:
- `GET /video_feed` → MJPEG stream  
- `GET /api/current_detection` → Latest detection JSON  
- `POST /api/order` → Store/order and optionally upload  

**OTA Rules**:
- Must fetch `model_version.json` at startup or intervals  
- Must validate downloaded model + matching `classes.json`  
- If mismatch occurs → reject model + rollback  

---

## 3.2 Next.js Frontend (App)

### Main Screen (`/`)
- Displays MJPEG stream  
- Shows candidate detections with stabilization  
- Operator can confirm/edit/remove items  
- Checkout triggers cloud or local order storage  

### Customer Screen (`/customer`)
- View-only  
- Mirrors cart + price summary  

### Dual-Screen Sync
- Implemented using `BroadcastChannel API`  
- Main screen is the single source of truth  

### Data Fetching & State
- SWR polling for `/api/current_detection`  
- React Context for global state  
- Static export (`output: 'export'`) served by FastAPI  

---

## 3.3 Cloud Config & Accounting

Cloud provides:
- `classes.json` — YOLO label definitions  
- `price_map.json` — Business mapping  
- `model_version.json` — OTA metadata  
- Order submission endpoint  

Cloud **does not**:
- Do inference  
- Serve as a runtime dependency  

System must remain fully functional offline.

---

# 4. Non-Goals

Smart-Canteen will not attempt to:

- Detect individual dishes or ingredients  
- Integrate payment SDKs during phase 1  
- Perform cloud-based inference  
- Synchronize carts/orders across multiple terminals  
- Store personal biometric data or faces  

---

# 5. Quality & UX Baseline

### 5.1 Performance Targets

Typical target hardware: J1900/i3 + USB 1080p

- Camera input: 720p preferred  
- Inference speed:  
  - **Target: ≥ 10 FPS**  
  - **Minimum acceptable: ≥ 5 FPS**  
- UI latency:  
  - **Target < 500 ms**  
  - Must not exceed 1 second  

### 5.2 Recognition Stability & Operator Control

- Detection must pass **N-frame stabilization** before adding to cart  
- Operator corrections override all automation  
- Frontend must visually flag unknown/missing mappings  

### 5.3 Offline Capability

System must:
- Load cached configs  
- Recognize and checkout offline  
- Queue orders locally  
- Upload backlog when network returns  

---

# 6. Architecture Guardrails

### 6.1 Monorepo Structure (Required)

```

/backend        # FastAPI + inference + OTA + static file serving
/frontend       # Next.js 14 static export (App Router)
/infra          # Packaging scripts, installer definitions, service configs

````

---

### 6.2 Backend Constraints

- Python 3.10+  
- FastAPI  
- YOLOv8n (OpenVINO or ONNX Runtime)  
- OpenCV  
- PyInstaller for EXE bundling  

Backend must:
- Own the camera  
- Never expose raw camera to browser  
- Guard configuration consistency  

---

### 6.3 Frontend Constraints

- Next.js 14+ App Router  
- shadcn/ui + Tailwind CSS  
- SWR for polling  
- React Context as global state  
- Static export only  
- No direct camera access (`getUserMedia` prohibited in production)  

---

## 6.4 Directory Structure (Simplified)

Smart-Canteen uses a mandatory monorepo layout.  
All source code must be placed in the following top-level directories:

```

/backend/      # Python backend: FastAPI service, YOLO inference, OTA, static file hosting
/frontend/     # Next.js 14 frontend: UI, customer/operator screens, static export
/infra/        # Build scripts, installers, packaging, deployment tools
/docs/         # Architecture notes, feature specs, operational documents (optional)

```

### Rules

- `backend` holds **all** inference logic, API endpoints, camera access, OTA logic, and local persistence.
- `frontend` holds **all** UI code and business rules (price mapping, stabilization, checkout logic).
- `infra` contains **only** scripts/configs for building, packaging, deployment.
- `docs` is recommended for specification and planning but optional.
- No code or config may exist outside these directories unless approved.





---

# 7. Configuration & Data Principles  
*(Shape-Oriented Classification)*

Smart-Canteen uses three tightly coupled configuration artifacts:

1. `classes.json` — shape-based model class labels  
2. `price_map.json` — mapping of shapes → names → prices  
3. `model_version.json` — OTA version metadata  

These must always remain consistent.

---

## 7.1 `classes.json` — Shape-Based YOLO Classes

Defines the **entire visual vocabulary** used by the inference model.

### Examples

```json
[
  "plate_round_small",
  "plate_round_large",
  "plate_square",
  "bowl_round_small",
  "bowl_round_large",
  "tray_rect_large"
]
````

### Mandatory Rules

* **No color-based labels**
  Colors are unstable under cafeteria lighting.

* **Labels describe only geometry and container type**
  (e.g., round/rectangular, small/large, plate/bowl/tray)

* **Array order must match YOLO class indices**

* **OTA model bundle must include the matching `classes.json`**
  If mismatch → reject + fallback

* **Backend must validate consistency** on startup and on OTA updates.

---

## 7.2 `price_map.json` — Mapping Shapes to Business Semantics

Keys: **MUST match exactly each label in `classes.json`**.

### Example

```json
{
  "plate_round_large": {
    "name": "Large Meal",
    "price": 15.0
  },
  "bowl_round_small": {
    "name": "Soup",
    "price": 4.0
  }
}
```

### Rules

* All keys must exist in `classes.json`

* Additional metadata allowed:

  * category
  * time_slots
  * discount_rules
  * enabled

* Missing or extra keys must raise warnings in the frontend

---

## 7.3 `model_version.json` — OTA Metadata

```json
{
  "version": "1.2.0",
  "url": "https://example.com/models/model_1.2.0.zip",
  "sha256": "checksum",
  "timestamp": "2024-03-01T12:00:00Z"
}
```

Rules:

* OTA must validate SHA256
* OTA must confirm model class count matches `classes.json`
* If inconsistency → reject update + fallback

---

## 7.4 Local Persistence Requirements

The backend must persist:

* Last valid `classes.json`
* Last valid `price_map.json`
* Last valid `model_version.json`
* Pending orders queue
* Logs

Persistence must survive crashes, reboot, and power loss.

---

## 7.5 Consistency Guarantees

| Component        | Source           | Must Match                 |
| ---------------- | ---------------- | -------------------------- |
| YOLO Model       | OTA bundle       | `classes.json` length      |
| Detection Labels | Backend runtime  | Strings in `classes.json`  |
| Pricing          | `price_map.json` | Keys ∈ `classes.json`      |
| UI               | Frontend         | Values in `price_map.json` |

If any mismatch:

1. Log error
2. Reject invalid config/model
3. Fallback to last valid state

---

# 8. Deployment & Runtime

* Final artifact: **single EXE** or minimal installer
* Installer must:

  * Deploy backend
  * Extract frontend static export
  * Register system service / autostart

### Runtime Sequence

1. Backend starts
2. Load model + verify `classes.json`
3. Serve frontend
4. Operator opens `/` on main screen
5. Customer screen displays `/customer`

---

# 9. Observability & Failure Handling

Backend must log:

* Startup version & config
* Camera connection
* Model loading
* OTA updates
* Order submission
* Errors & fallback decisions

Failures:

* **Camera offline** → UI shows clear error
* **Bad model** → fallback to previous
* **Cloud unreachable** → cached config + local queue

---

# 10. Privacy & Compliance

* No face or personal data
* No video recording
* Frames exist only in memory
* Orders avoid sensitive identifiers

---

# 11. Collaboration Rules

* All new features require a written `spec`
* Deviations from constitution must be documented in PR
* Documentation must include:

  * Installation
  * Maintenance
  * Troubleshooting
  * Updating model & frontend instructions

---

# 12. Constitution Amendments

* Require written proposal
* Must be approved by:

  * One lead engineer
  * One ops/admin stakeholder
* Updated via PR referencing proposal

```
```
