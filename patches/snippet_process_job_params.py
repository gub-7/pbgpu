# INSERT these 2 lines in process_job(), in the parameters section,
# right AFTER the line:
#     texture_resolution = int(params.get("texture_resolution", 1024))
# and BEFORE the line:
#     chunk_size = int(params.get("chunk_size", 8192))

            reproject_colors = bool(params.get("reproject_colors", True))
            projection_axis = params.get("projection_axis", "auto")

