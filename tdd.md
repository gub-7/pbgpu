I reviewed the uploaded project. Right now it is a **solid single-image reconstruction service**, not a true few-view reconstruction system.

## Current state of the project

What it already does well:

* FastAPI job API, Redis-backed queueing, storage, preview generation, and worker split are already in place.
* Good **single-image preprocessing** stack: segmentation, alpha hardening, TripoSR-oriented framing, preview stages, and model-specific worker paths.
* Two model lanes exist:

  * **TripoSR** for fast single-image mesh generation.
  * **TRELLIS.2** for higher-quality single-image asset generation.

What it is missing for your target:

* No concept of a **multi-view capture set**. The API accepts one file, not 5 canonical views.
* No **view bookkeeping** for front/back/left/right/top.
* No **cross-view consistency** stage.
* No **camera pose estimation / canonical camera prior**.
* No true **few-view reconstruction backbone**.
* No **mesh + Gaussian refinement** stage.
* No **fusion / completion / retexturing** pipeline.
* Tests are mostly around API shape and image preprocessing, not reconstruction quality.

So the current project is a good shell, but the actual reconstruction core is still **single-view oriented**.

## Best target architecture for 5 canonical views

For input views:

* front
* back
* left
* right
* top

the best practical pipeline today is **not** “run TripoSR 5 times.”
The right direction is:

1. **Canonical multi-view ingestion**
2. **Per-view segmentation + matting**
3. **Cross-view normalization and consistency checks**
4. **Camera initialization from known canonical views**
5. **Few-view reconstruction backbone**
6. **Joint mesh/Gaussian refinement**
7. **Generative completion only where evidence is weak**
8. **Texture/material bake**
9. **GLB export + previews + diagnostics**

Why this direction:

* TRELLIS.2 is strong as a high-fidelity 3D generative prior with arbitrary topology and PBR output. ([Microsoft GitHub][1])
* MVSplat is a strong sparse multi-view Gaussian approach designed specifically for sparse multi-view images. ([Donny Y. Chen][2])
* The frontier is moving toward **joint mesh + Gaussian optimization** rather than “splat first, mesh later.” OMeGa is a good signal for that direction. ([CVF Open Access][3])
* Hunyuan3D 2.x is relevant as a completion / fallback / comparison path for multi-view-conditioned asset generation. ([GitHub][4])

## Recommendation: the pipeline you should build

For this specific product, I would make the **primary path**:

**5-view canonical input → segmentation/cleanup → canonical camera solve → sparse multi-view Gaussian initialization → joint mesh/Gaussian refinement → TRELLIS.2-guided completion → texture baking → GLB**

And keep these as secondary paths:

* **Fast mode:** 5-view → Hunyuan/TRELLIS-style direct multi-view generation
* **Fallback mode:** best single image → TRELLIS.2 only
* **Debug mode:** per-view single-image reconstruction for diagnostics only

---

# TDD: Few-View 3D Reconstruction from 5 Canonical Views

## 1. Title

**Technical Design Document: Canonical 5-View 3D Reconstruction Pipeline**

## 2. Objective

Implement a production-grade pipeline that reconstructs a high-quality 3D asset from 5 canonical input images:

* front
* back
* left
* right
* top

The system should output:

* textured GLB
* intermediate previews
* diagnostic QA metrics
* optional Gaussian preview/debug artifact
* job progress and reproducible metadata

## 3. Success criteria

Primary:

* reconstruct a coherent watertight or near-watertight mesh from 5 canonical views
* preserve subject identity across views
* avoid major hallucinated asymmetry
* produce a clean exportable GLB

Quality targets:

* no missing torso/head-scale regions for rigid or mostly rigid subjects
* no large floating geometry fragments
* consistent silhouette agreement with all 5 inputs
* deterministic output under fixed seed + same inputs

Operational targets:

* async queued jobs
* resumable stages
* stage-level artifacts saved to storage
* worker crash isolation
* model lane selection by job config

## 4. Non-goals

Not in v1:

* full turntable video ingestion
* articulated dynamic subjects
* high-quality fur strand reconstruction
* animation-ready rigging
* exact bottom-side recovery without evidence
* photogrammetry-grade metrology

## 5. Product assumptions

Assume:

* subject is centered
* each image is reasonably sharp
* views are approximately canonical
* background can be segmented
* subject is mostly static and consistent across views

Optional v1.1:

* allow 6th view: bottom or front-3/4
* allow arbitrary sparse views with pose estimation

---

## 6. Input contract

### Request

`POST /api/upload_multiview`

Form-data:

* `front`: image
* `back`: image
* `left`: image
* `right`: image
* `top`: image
* `category`
* `pipeline`
* `params` JSON

### New enums

`PipelineEnum`

* `canonical_mv_hybrid`
* `canonical_mv_fast`
* `canonical_mv_generative`
* `singleview_triposr`
* `singleview_trellis2`

### New params model

`CanonicalMVParams`

* `output_resolution`
* `mesh_resolution`
* `texture_resolution`
* `use_joint_refinement`
* `use_trellis_completion`
* `use_hunyuan_completion`
* `symmetry_prior`
* `category_prior`
* `generate_debug_renders`
* `generate_gaussian_debug`
* `decimation_target`
* `seed`

---

## 7. High-level architecture

### New stages

1. upload validation
2. per-view preprocessing
3. view consistency
4. camera initialization
5. coarse reconstruction
6. joint refinement
7. completion
8. mesh extraction
9. texture baking
10. mesh cleanup/export
11. QA scoring

### New job stage model

Replace current simple:

* preprocessing
* generation

with:

* ingest
* preprocess_views
* validate_views
* initialize_cameras
* reconstruct_coarse
* refine_joint
* complete_geometry
* bake_texture
* export
* qa

---

## 8. Current codebase impact

## 8.1 API layer

Current `/api/upload` is single-file only.

### Changes

Add:

* `/api/upload_multiview`
* `/api/job/{job_id}/artifacts`
* `/api/job/{job_id}/metrics`
* `/api/job/{job_id}/rerun_stage`
* `/api/job/{job_id}/download/{artifact}`

### Data model change

Store per-view metadata:

```json
{
  "views": {
    "front": {"filename":"front.png","status":"ok"},
    "back": {"filename":"back.png","status":"ok"},
    "left": {"filename":"left.png","status":"ok"},
    "right": {"filename":"right.png","status":"ok"},
    "top": {"filename":"top.png","status":"ok"}
  }
}
```

## 8.2 Storage layer

Current storage assumes one upload and one output.

### Changes

Per job:

* `uploads/{job_id}/views/*.png`
* `previews/{job_id}/raw/{view}.png`
* `previews/{job_id}/segmented/{view}.png`
* `previews/{job_id}/normalized/{view}.png`
* `artifacts/{job_id}/camera_init.json`
* `artifacts/{job_id}/coarse_gaussians.ply`
* `artifacts/{job_id}/coarse_mesh.glb`
* `artifacts/{job_id}/refined_mesh.glb`
* `artifacts/{job_id}/textures/*`
* `artifacts/{job_id}/metrics.json`
* `outputs/{job_id}/final.glb`

## 8.3 Worker layer

Current workers are model-specific and single-image.

### Changes

Create a new worker:

* `workers/canonical_mv_worker.py`

Submodules:

* `workers/stages/ingest.py`
* `workers/stages/preprocess_views.py`
* `workers/stages/view_consistency.py`
* `workers/stages/camera_init.py`
* `workers/stages/coarse_reconstruction.py`
* `workers/stages/joint_refinement.py`
* `workers/stages/completion.py`
* `workers/stages/texturing.py`
* `workers/stages/export.py`
* `workers/stages/qa.py`

---

## 9. Detailed pipeline design

## 9.1 Per-view preprocessing

Use your existing segmentation/preprocessing stack as a base, but adapt it for multi-view.

### Required changes

* process each canonical image separately
* unify crop scale across views
* unify background convention across views
* normalize lighting and white balance across views
* compute silhouette masks per view
* compute contour descriptors for each view

### New outputs per view

* RGBA segmented image
* binary mask
* bbox
* centroid
* silhouette polygon
* color histogram
* sharpness score

### Important addition

Current preprocessing is optimized for single-image TripoSR framing.
For few-view, framing must become **cross-view consistent**, not per-image independently.

Implement:

* global scale target from median subject extent
* shared canvas size
* shared canonical center convention

---

## 9.2 View consistency validation

This is new and important.

### Goals

Catch bad input early:

* wrong side uploaded in wrong slot
* mirrored image
* severe segmentation failure
* inconsistent scale
* subject deformation
* extreme lighting mismatch

### Checks

* silhouette area consistency across front/back and left/right
* left-right mirror plausibility
* front/back shape mismatch score
* top view overlap plausibility
* CLIP/DINO embedding identity consistency
* segmentation confidence thresholds

### Result

* hard fail on catastrophic issues
* soft warnings on moderate issues
* store metrics for frontend review

---

## 9.3 Camera initialization

Because the views are canonical, do not start with arbitrary SfM as the default.

### v1 strategy

Use a fixed canonical camera rig:

* front = yaw 0
* right = yaw 90
* back = yaw 180
* left = yaw 270
* top = pitch -90

All with shared focal prior and subject-centered origin.

Then optionally refine:

* translation scale
* focal length
* small rotational offsets

### v2 strategy

Add learned or optimization-based camera refinement from silhouettes and photometric agreement.

This is much more stable for your exact input mode than pretending the 5 views are random internet photos.

---

## 9.4 Coarse reconstruction backbone

This is the first major new core.

### Recommended v1

Implement a **sparse multi-view Gaussian backbone** inspired by MVSplat-style reasoning:

* consume the 5 posed images
* predict coarse depth / occupancy / Gaussian anchors
* build initial Gaussian scene
* optionally derive initial coarse mesh

Why:

* sparse multi-view Gaussian reconstruction is much closer to your real input than single-image TripoSR. ([Donny Y. Chen][2])

### Practical implementation choices

Option A, best:

* integrate an existing sparse multi-view Gaussian model

Option B, safer engineering:

* initialize from silhouette visual hull + learned depth priors
* build Gaussian cloud from fused depth/mask estimates
* optimize photometrically

For your codebase, **Option B is easier to ship first**, then swap the coarse initializer later.

### v1 coarse result artifacts

* `coarse_voxel.npz`
* `coarse_visual_hull_mesh.glb`
* `coarse_gaussians.ply`
* `coarse_depth_{view}.png`

---

## 9.5 Joint mesh + Gaussian refinement

This is the most important design choice.

Instead of:

* reconstruct splats
* extract mesh at the end
* hope it works

do:

* keep a mesh and Gaussian representation active together
* optimize both under view reconstruction loss and geometry regularizers

That direction matches where joint optimization methods like OMeGa are going. ([CVF Open Access][3])

### Optimization losses

* photometric reconstruction loss
* silhouette loss
* normal consistency loss
* Laplacian smoothness
* edge-length regularization
* symmetry regularization when enabled
* top-view coverage prior
* view agreement / reprojection consistency

### Why this matters

Your subject has only 5 views. If you wait until the end to get a mesh, you will amplify ambiguity. Joint refinement constrains the geometry earlier.

---

## 9.6 Generative completion

Use generative models only where observation is weak.

### Recommended order

1. preserve evidence-backed geometry
2. identify weak-confidence regions
3. run completion prior only on weak-confidence regions
4. fuse completed geometry back into refined mesh

### Recommended priors

* TRELLIS.2 as primary completion prior
* optional Hunyuan3D path for comparison / fallback

TRELLIS.2 is strong for high-fidelity image-to-3D with topology/material flexibility. ([Microsoft GitHub][1])
Hunyuan3D 2.x is also relevant for high-resolution shape + texture generation and can serve as a secondary prior or A/B path. ([GitHub][4])

### Fusion policy

Do not replace the entire mesh with generative output.
Use:

* confidence mask over surface regions
* patch replacement or displacement-field correction
* optional back-side fill only

---

## 9.7 Texture/material bake

### Inputs

* refined mesh
* calibrated views
* per-view masks
* visibility maps

### Steps

* unwrap UVs
* compute visibility for each texel
* project colors from best visible views
* blend with seam-aware weighting
* inpaint occluded texels
* optional PBR estimation

### Per-texel weights

* viewing angle
* segmentation confidence
* local sharpness
* distance to silhouette boundary
* cross-view color consistency

---

## 9.8 Mesh cleanup/export

### Required operations

* remove floaters
* hole fill with confidence threshold
* decimation
* normal recompute
* self-intersection checks
* manifoldness checks
* scale normalization
* GLB export

### Optional

* alternate exports: OBJ, PLY, USDZ
* turntable preview render
* wireframe preview

---

## 9.9 QA and scoring

Current project lacks true reconstruction QA.

### Add metrics

* silhouette IoU per view
* masked PSNR/SSIM/LPIPS per view
* mesh component count
* watertightness
* self-intersection count
* texture seam score
* camera reprojection error
* symmetry deviation score
* completion coverage ratio

### Overall quality gate

Produce:

* `quality_score`
* `warnings[]`
* `recommended_retry[]`

Example:

```json
{
  "quality_score": 0.84,
  "warnings": ["top_view_segmentation_low_confidence"],
  "recommended_retry": ["re-upload sharper top image"]
}
```

---

## 10. Suggested repository refactor

```text
gpu-cluster/
  api/
    main.py
    models.py
    job_manager.py
    storage.py
  pipelines/
    canonical_mv/
      __init__.py
      config.py
      orchestrator.py
      ingest.py
      preprocess.py
      consistency.py
      camera_init.py
      coarse_recon.py
      joint_refine.py
      completion.py
      texturing.py
      export.py
      qa.py
  workers/
    canonical_mv_worker.py
    triposr_worker.py
    trellis2_worker.py
  vision/
    segmentation/
    geometry/
    cameras/
    texturing/
    metrics/
  tests/
    unit/
    integration/
    regression/
    fixtures/
```

---

## 11. API design

## 11.1 Upload

`POST /api/upload_multiview`

Response:

```json
{
  "job_id": "uuid",
  "status": "queued",
  "pipeline": "canonical_mv_hybrid",
  "views_received": ["front","back","left","right","top"]
}
```

## 11.2 Status

`GET /api/job/{job_id}/status`

Add:

* `pipeline`
* `current_stage`
* `stage_progress`
* `warnings`
* `artifacts_available`

## 11.3 Metrics

`GET /api/job/{job_id}/metrics`

## 11.4 Artifacts

`GET /api/job/{job_id}/artifacts`

Returns URLs for:

* segmented views
* masks
* depth previews
* coarse mesh
* final mesh
* turntable

---

## 12. Testing strategy

## 12.1 Unit tests

* per-view validation
* canonical camera initialization
* view ordering checks
* silhouette consistency scoring
* UV baking weights
* mesh cleanup rules

## 12.2 Integration tests

* full 5-view happy path
* missing top view
* swapped left/right
* mirrored back image
* segmentation failure in one view
* low-memory worker retry behavior

## 12.3 Regression tests

Create a small fixture suite:

* rigid object
* bust
* animal-like organic object
* reflective object
* thin-structure object

For each, store expected bounds:

* silhouette IoU
* component count
* face count range
* runtime range

## 12.4 Quality benchmarks

Add offline benchmark script:

* runs fixed fixture set
* exports metrics table
* compares against previous commit
* fails CI if quality regresses beyond threshold

---

## 13. Phased implementation plan

## Phase 0: Foundation

* add multi-view upload API
* add storage layout for views
* add new job stages
* add preview support per view

## Phase 1: Multi-view preprocessing

* reuse existing segmentation
* add cross-view framing
* add validation and consistency metrics

## Phase 2: Canonical camera rig

* fixed canonical extrinsics
* focal/scale refinement
* reprojection utilities

## Phase 3: Coarse geometry

* visual hull / fused depth initializer
* coarse mesh and Gaussian outputs

## Phase 4: Joint refinement

* photometric + silhouette optimization
* active mesh + Gaussian refinement

## Phase 5: Generative completion

* TRELLIS.2 completion on low-confidence regions
* optional Hunyuan comparison path

## Phase 6: Texture + export

* UV unwrap
* multi-view texture projection
* GLB export
* QA scoring

## Phase 7: Benchmarking and tuning

* fixture suite
* regression dashboard
* parameter presets per category

---

## 14. Concrete engineering recommendation

If this were my project, I would implement in this order:

1. **Do not touch TripoSR worker first**
2. Build a totally separate `canonical_mv_hybrid` path
3. Reuse only:

   * API/job/storage scaffolding
   * segmentation utilities
   * preview generation
4. Add:

   * multi-view API
   * canonical rig
   * visual hull/depth fusion initializer
   * joint mesh/Gaussian refinement
   * TRELLIS.2 regional completion
   * texture bake

That is the cleanest path because the current code is strongly shaped around single-image assumptions.

## 15. Blunt assessment

The project is currently at about:

* **80% of the surrounding service infrastructure**
* **60% of useful single-image preprocessing**
* **25% of what you need for a true few-view reconstruction system**
* **0–10% of the actual sparse multi-view reconstruction core**

So the good news is: the shell is there.
The missing part is the important part.

## 16. Best possible pipeline, in one sentence

**Use the existing service shell, replace the single-image assumption with a canonical 5-view job model, reconstruct a coarse subject from fixed-view geometry, refine with joint mesh/Gaussian optimization, and use TRELLIS.2 only as a localized completion/material prior rather than the main source of truth.**

If you want, I’ll turn this into a **file-by-file implementation backlog with exact new classes, endpoints, and function signatures**.

[1]: https://microsoft.github.io/TRELLIS.2/?utm_source=chatgpt.com "TRELLIS.2: Native and Compact Structured Latents for 3D ..."
[2]: https://donydchen.github.io/mvsplat/?utm_source=chatgpt.com "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi ..."
[3]: https://openaccess.thecvf.com/content/WACV2026/html/Cao_OMeGa_Joint_Optimization_of_Explicit_Meshes_and_Gaussian_Splats_for_WACV_2026_paper.html?utm_source=chatgpt.com "WACV 2026 Open Access Repository"
[4]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2?utm_source=chatgpt.com "Tencent-Hunyuan/Hunyuan3D-2: High-Resolution 3D ..."

