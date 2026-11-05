import mujoco

model = mujoco.MjModel.from_xml_path("assets/mjcf/arena.xml")
#mujoco.mj_saveLastXML("arena_compile.xml", model)   # modèle fusionné complet

sol = model.geom("floor")
print("rgba :", sol.rgba, "| matériau :", sol.matid, "| groupe :", sol.group)