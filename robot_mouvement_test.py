import mujoco
import numpy as np


# Ligne droite : v = r·ω = 0,25 m/s
#print("position :", data.qpos[:3], "vitesse x :", data.qvel[0], "(attendu 0.25)")

# Rotation sur place : ω = r(ωR − ωL)/L
#mujoco.mj_resetData(model, data)
#data.ctrl[:] = [-5, 5]                     # roue gauche en arrière, droite en avant
#print("vitesse de lacet :", data.qvel[5], "(attendu", r * 10 / L, ")")
#print("lacet final :", lacet(data.qpos[3:7]))

import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("assets/mjcf/arena.xml")
data = mujoco.MjData(model)

def commande(t):
    """Enchaîne trois manœuvres de 2 s, en boucle."""
    phase = t % 6
    if phase < 2:
        return [5, 5]      # ligne droite
    elif phase < 4:
        return [-5, 5]     # rotation sur place (sens trigonométrique)
    else:
        return [3, 6]      # virage à gauche

with mujoco.viewer.launch_passive(model, data) as viewer:
    with viewer.lock():
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("chassis").id
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -30

    while viewer.is_running():
        t0 = time.time()
        data.ctrl[:] = commande(data.time)
        mujoco.mj_step(model, data)
        #print("vitesse de lacet :", data.qvel[5], "(attendu", r * 10 / L, ")")
        #print("lacet final :", lacet(data.qpos[3:7]))
        viewer.sync()
        reste = model.opt.timestep - (time.time() - t0)
        if reste > 0:
            time.sleep(reste)