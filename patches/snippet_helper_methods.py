    def _compute_robust_vertex_normals(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute per-vertex normals robustly, handling degenerate meshes.
        Falls back to a default direction if trimesh normals fail.
        """
        try:
            normals = np.asarray(mesh.vertex_normals)
            if normals.shape[0] == len(mesh.vertices):
                # Check for NaN/zero normals
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                bad = (norms.squeeze() < 1e-8) | ~np.isfinite(normals).all(axis=1)
                if bad.any():
                    logger.warning(
                        "Reprojection: %d/%d vertex normals are degenerate, "
                        "replacing with [0,0,1]",
                        int(bad.sum()), len(normals),
                    )
                    normals[bad] = [0.0, 0.0, 1.0]
                    norms[bad.reshape(-1)] = 1.0
                # Re-normalize
                normals = normals / np.maximum(norms, 1e-8)
                return normals
        except Exception as e:
            logger.warning("Reprojection: vertex_normals failed: %s", e)

        # Fallback: all normals point toward +Z (front)
        logger.warning(
            "Reprojection: using fallback normals [0,0,1] for all vertices",
        )
        return np.tile([0.0, 0.0, 1.0], (len(mesh.vertices), 1))

    def _auto_detect_forward_axis(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        img_rgb: np.ndarray,
        fg_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Auto-detect the forward axis by trying all 6 axis directions and
        picking the one where the most front-facing vertices project onto
        foreground pixels in the source image.
        """
        H, W = fg_mask.shape
        candidates = [
            ("+z", np.array([0.0, 0.0, 1.0])),
            ("-z", np.array([0.0, 0.0, -1.0])),
            ("+x", np.array([1.0, 0.0, 0.0])),
            ("-x", np.array([-1.0, 0.0, 0.0])),
            ("+y", np.array([0.0, 1.0, 0.0])),
            ("-y", np.array([0.0, -1.0, 0.0])),
        ]

        best_name = "+z"
        best_axis = candidates[0][1]
        best_hits = -1

        for name, axis in candidates:
            dots = normals @ axis
            front = dots > 0.1
            if front.sum() == 0:
                continue

            fwd_dim = int(np.argmax(np.abs(axis)))
            fwd_sign = float(axis[fwd_dim])
            plane_dims = [d for d in range(3) if d != fwd_dim]
            d0, d1 = plane_dims

            fv = vertices[front]
            u = (fv[:, d0] + _TRIPOSR_COORD_HALF) / _TRIPOSR_COORD_EXTENT
            v = (_TRIPOSR_COORD_HALF - fv[:, d1]) / _TRIPOSR_COORD_EXTENT
            if fwd_sign < 0:
                u = 1.0 - u

            px = np.clip((u * (W - 1)).astype(np.int32), 0, W - 1)
            py = np.clip((v * (H - 1)).astype(np.int32), 0, H - 1)

            hits = int(fg_mask[py, px].sum())
            logger.info(
                "Axis probe %s: %d front verts, %d fg hits",
                name, int(front.sum()), hits,
            )

            if hits > best_hits:
                best_hits = hits
                best_axis = axis
                best_name = name

        logger.info(
            "Auto-detected forward axis: %s (%d foreground hits)",
            best_name, best_hits,
        )
        return best_axis

    @staticmethod
    def _parse_axis_string(axis_str: str) -> np.ndarray:
        """Parse an axis string like '+z', '-x' into a unit vector."""
        axis_map = {
            "+x": [1, 0, 0], "-x": [-1, 0, 0],
            "+y": [0, 1, 0], "-y": [0, -1, 0],
            "+z": [0, 0, 1], "-z": [0, 0, -1],
        }
        key = axis_str.lower().strip()
        if key in axis_map:
            return np.array(axis_map[key], dtype=float)
        logger.warning("Unknown axis '%s', defaulting to +z", axis_str)
        return np.array([0.0, 0.0, 1.0])

