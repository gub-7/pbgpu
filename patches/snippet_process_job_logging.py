# INSERT these 2 lines in process_job(), right AFTER the block:
#     logger.info(
#         f"Parameters: foreground_ratio={foreground_ratio}, "
#         f"mc_resolution={mc_resolution}, mc_resolution_final={mc_resolution_final}, "
#         f"texture_res={texture_resolution}, "
#         f"user_mc_threshold={user_mc_threshold}, chunk_size={chunk_size}, "
#         f"export_mode={export_mode}, bake_texture={bake_texture}"
#     )
# and BEFORE:
#     self.load_model()

            logger.info(f"Reprojection: reproject_colors={reproject_colors}, "
                        f"projection_axis={projection_axis}")

