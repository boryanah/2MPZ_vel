from pixell import enmap, utils, powspec, enplot, reproject #, pointsrcs

base_mask = enmap.read_map("/mnt/marvin1/boryanah/2MPZ_vel/cmb_data/wide_mask_GAL070_apod_1.50_deg_wExtended.fits")
point_mask = enmap.read_map("/mnt/marvin1/boryanah/2MPZ_vel/cmb_data/act_pts_mask_fluxcut_15.0mJy_at150Ghz_rad_5.0_monster_dust.fits")
print(base_mask.wcs, base_mask.shape)
print(point_mask.wcs, point_mask.shape)

final_mask = base_mask * point_mask
enmap.write_map("/mnt/marvin1/boryanah/2MPZ_vel/cmb_data/wide_mask_GAL070_apod_1.50_deg_wExtended_srcfree_Pato.fits", final_mask)
