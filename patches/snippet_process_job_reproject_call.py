# INSERT this block in process_job(), right AFTER:
#     logger.info("Scene codes deleted and GPU cache cleared")
# and BEFORE:
#     self.job_manager.update_job(job_id, progress=85)

                # ----------------------------------------------------------
                # Step 6: Reproject input image colors onto mesh
                # ----------------------------------------------------------
                if reproject_colors:
                    logger.info("Reprojecting input image colors onto mesh...")
                    try:
                        mesh = self._reproject_image_onto_mesh(
                            mesh=mesh,
                            source_image_path=str(final_image_path),
                            projection_axis=projection_axis,
                        )
                        _color_health("post_reproject", mesh)
                        logger.info(
                            "Image reprojection complete: %d verts, %d faces",
                            len(mesh.vertices), len(mesh.faces),
                        )
                    except Exception as e:
                        logger.warning(
                            "Image reprojection failed: %s. "
                            "Falling back to TripoSR native vertex colors.",
                            e, exc_info=True,
                        )

