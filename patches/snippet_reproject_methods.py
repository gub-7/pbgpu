    # ------------------------------------------------------------------
    # Image reprojection — project source image onto mesh surface
    # ------------------------------------------------------------------

    def _reproject_image_onto_mesh(
        self,
        mesh: trimesh.Trimesh,
        source_image_path: str,
        projection_axis: str = "auto",
        front_threshold: float = 0.1,
        background_alpha_threshold: int = 128,
    ) -> trimesh.Trimesh:
        """
        Replace TripoSR's low-quality vertex colors with colors sampled
        directly from the source image via orthographic reprojection.

        TripoSR's ``extract_mesh(has_vertex_color=True)`` produces a
        low-frequency, smeared color field that does not faithfully map
        the input image's appearance.  This method fixes that by:

        1. Loading the source image (with alpha for foreground masking).
        2. Auto-detecting (or using a specified) forward axis.
        3. Computing per-vertex normals to identify front-facing vertices.
        4. Projecting front-facing vertices into image UV space and
           sampling the source image color at each projected position.
        5. Filling back-facing / background vertices via KDTree
           nearest-neighbor from successfully-projected front vertices.

        Args:
            mesh: The extracted trimesh with geometry.
            source_image_path: Path to the preprocessed source image
                (``final.png``).  Loaded as RGBA if possible for
                foreground masking; falls back to RGB.
            projection_axis: One of ``"auto"``, ``"+z"``, ``"-z"``,
                ``"+x"``, ``"-x"``, ``"+y"``, ``"-y"``.
            front_threshold: Dot-product threshold for front-facing
                normals (vertices with ``dot(normal, forward) > threshold``
                are considered front-facing).
            background_alpha_threshold: Alpha value below which a pixel
                is considered background (only used when the image has
                an alpha channel).

        Returns:
            The mesh with reprojected vertex colors applied.
        """
        vertices = np.asarray(mesh.vertices)
        n_verts = len(vertices)
        if n_verts == 0:
            logger.warning("Reprojection: mesh has no vertices, skipping.")
            return mesh

        # ---- Load source image (RGBA preferred for foreground mask) ----
        raw_img = Image.open(source_image_path)
        has_alpha = raw_img.mode in ("RGBA", "LA", "PA")
        img_rgba = raw_img.convert("RGBA")
        img_arr = np.asarray(img_rgba)  # (H, W, 4) uint8
        H, W = img_arr.shape[:2]
        img_rgb = img_arr[:, :, :3]
        img_alpha = img_arr[:, :, 3]

        # Build foreground mask
        if has_alpha:
            fg_mask = img_alpha >= background_alpha_threshold
        else:
            # No alpha channel — assume all pixels are foreground
            fg_mask = np.ones((H, W), dtype=bool)

        fg_count = int(fg_mask.sum())
        logger.info(
            "Reprojection: image %dx%d, has_alpha=%s, fg_pixels=%d/%d (%.1f%%)",
            W, H, has_alpha, fg_count, H * W,
            100.0 * fg_count / max(1, H * W),
        )

        # ---- Compute vertex normals ----
        normals = self._compute_robust_vertex_normals(mesh)

        # ---- Determine forward axis ----
        if projection_axis == "auto":
            forward_axis = self._auto_detect_forward_axis(
                vertices, normals, img_rgb, fg_mask,
            )
        else:
            forward_axis = self._parse_axis_string(projection_axis)

        logger.info("Reprojection: using forward axis = %s", forward_axis)

        # ---- Identify front-facing vertices ----
        dots = normals @ forward_axis
        front_mask = dots > front_threshold
        n_front = int(front_mask.sum())
        logger.info(
            "Reprojection: %d/%d vertices are front-facing (threshold=%.2f)",
            n_front, n_verts, front_threshold,
        )

        if n_front == 0:
            logger.warning(
                "Reprojection: no front-facing vertices found! "
                "Trying with relaxed threshold (0.0)...",
            )
            front_mask = dots > 0.0
            n_front = int(front_mask.sum())
            if n_front == 0:
                logger.warning(
                    "Reprojection: still no front-facing vertices. "
                    "Skipping reprojection.",
                )
                return mesh

        # ---- Project front-facing vertices to image UV ----
        abs_fwd = np.abs(forward_axis)
        fwd_dim = int(np.argmax(abs_fwd))
        plane_dims = [d for d in range(3) if d != fwd_dim]
        fwd_sign = float(forward_axis[fwd_dim])
        d0, d1 = plane_dims

        front_verts = vertices[front_mask]

        u_coords = (front_verts[:, d0] + _TRIPOSR_COORD_HALF) / _TRIPOSR_COORD_EXTENT
        v_coords = (_TRIPOSR_COORD_HALF - front_verts[:, d1]) / _TRIPOSR_COORD_EXTENT

        # Flip u if forward axis is negative (mirror correction)
        if fwd_sign < 0:
            u_coords = 1.0 - u_coords

        # Convert UV to pixel coordinates
        px = np.clip((u_coords * (W - 1)).astype(np.int32), 0, W - 1)
        py = np.clip((v_coords * (H - 1)).astype(np.int32), 0, H - 1)

        # ---- Sample colors and apply foreground mask ----
        sampled_rgb = img_rgb[py, px]  # (n_front, 3)
        sampled_fg = fg_mask[py, px]   # (n_front,) bool

        n_fg_hit = int(sampled_fg.sum())
        logger.info(
            "Reprojection: %d/%d front verts project to foreground pixels",
            n_fg_hit, n_front,
        )

        # ---- Build final color array ----
        # Start with existing vertex colors as fallback (or gray)
        if (
            hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
        ):
            existing_vc = np.asarray(mesh.visual.vertex_colors)
            if existing_vc.shape[0] == n_verts:
                final_colors = existing_vc[:, :3].copy()
                if np.issubdtype(final_colors.dtype, np.floating):
                    if final_colors.max() <= 1.0:
                        final_colors = (final_colors * 255).astype(np.uint8)
                    else:
                        final_colors = np.clip(final_colors, 0, 255).astype(np.uint8)
                else:
                    final_colors = np.clip(final_colors, 0, 255).astype(np.uint8)
            else:
                final_colors = np.full((n_verts, 3), 128, dtype=np.uint8)
        else:
            final_colors = np.full((n_verts, 3), 128, dtype=np.uint8)

        # Write reprojected colors for front-facing + foreground vertices
        good_mask_local = sampled_fg  # within front_mask subset
        front_indices = np.where(front_mask)[0]
        good_global_indices = front_indices[good_mask_local]

        final_colors[good_global_indices] = np.clip(
            sampled_rgb[good_mask_local], 0, 255,
        ).astype(np.uint8)

        n_good = len(good_global_indices)
        logger.info("Reprojection: %d vertices got direct image colors", n_good)

        # ---- Fill remaining vertices from nearest good vertex ----
        if n_good > 0 and n_good < n_verts:
            bad_mask = np.ones(n_verts, dtype=bool)
            bad_mask[good_global_indices] = False
            n_bad = int(bad_mask.sum())

            logger.info(
                "Reprojection: filling %d remaining vertices via "
                "nearest-neighbor from %d good vertices...",
                n_bad, n_good,
            )

            good_verts = vertices[good_global_indices]
            bad_verts = vertices[bad_mask]
            tree = cKDTree(good_verts)
            _, nn_indices = tree.query(bad_verts)
            final_colors[bad_mask] = final_colors[good_global_indices[nn_indices]]

            logger.info("Reprojection: nearest-neighbor fill complete")

        # ---- Apply to mesh ----
        alpha_col = np.full((n_verts, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([final_colors, alpha_col], axis=1)

        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=rgba,
        )

        logger.info(
            "Reprojection complete: applied %d direct + %d filled colors "
            "to %d vertices",
            n_good, n_verts - n_good, n_verts,
        )
        return mesh

