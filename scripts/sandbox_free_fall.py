import mediapy as media
import imageio_ffmpeg
import matplotlib.pyplot as plt
import mujoco
import time
import itertools
import numpy as np
import mujoco.viewer
import matplotlib.pyplot as plt

# Indique à mediapy où se trouve l'exécutable ffmpeg
media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

free_body_MJCF = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
    rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"
    reflectance=".2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1" mode="trackcom"/>
    <geom name="floor" type="plane" pos="0 0 -.5" size="2 2 .1" material="grid" solimp=".99 .99 .01" solref=".001 1"/>
    <body name="box" pos="0 0 2.45">
      <freejoint/>
      <geom name="red_box" type="box" size=".1 .1 .1" rgba="1 0 0 1" solimp=".99 .99 .01"  solref=".001 1"/>
      <camera name="fixed" pos="0 -.6 .3" xyaxes="1 0 0 0 1 2"/>
      <camera name="track" pos="0 -.6 .3" xyaxes="1 0 0 0 1 2" mode="track"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(free_body_MJCF)
data = mujoco.MjData(model)
height = 400
width = 600

with mujoco.Renderer(model, height, width) as renderer:
  mujoco.mj_forward(model, data)
  renderer.update_scene(data, "fixed")

  media.show_image(renderer.render())

n_frames = 200
height = 240
width = 320
frames = []

# visualize contact frames and forces, make body transparent
options = mujoco.MjvOption()
mujoco.mjv_defaultOption(options)
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

# tweak scales of contact visualization elements
model.vis.scale.contactwidth = 0.1
model.vis.scale.contactheight = 0.03
model.vis.scale.forcewidth = 0.05
model.vis.map.force = 0.3

mujoco.mj_resetData(model, data)
#data.qvel[3:6] = 5 * np.random.randn(3)

with mujoco.viewer.launch_passive(model, data) as viewer:
    with viewer.lock():
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        # caméra "track" définie dans le MJCF
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera("track").id

    while viewer.is_running():
        t0 = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        # cadence temps réel
        reste = model.opt.timestep - (time.time() - t0)
        if reste > 0:
            time.sleep(reste)

#script de visualisation : 

def chute(dt, integrateur=mujoco.mjtIntegrator.mjINT_EULER, duree=0.5):
    """Simule la chute et renvoie les temps et les hauteurs du centre."""
    model = mujoco.MjModel.from_xml_string(free_body_MJCF)
    model.opt.timestep = dt
    model.opt.integrator = integrateur
    data = mujoco.MjData(model)          # qpos initial = pos du body

    n = int(round(duree / dt))
    t = np.empty(n + 1)
    z = np.empty(n + 1)
    t[0], z[0] = data.time, data.qpos[2]
    for k in range(1, n + 1):
        mujoco.mj_step(model, data)
        if data.ncon > 0:
            raise RuntimeError(f"Contact à t = {data.time:.3f} s")
        t[k], z[k] = data.time, data.qpos[2]
    return t, z, -model.opt.gravity[2]

z0 = 2.45
for dt in (2e-3, 1e-3):
    t, z, g = chute(dt)
    n = np.arange(len(t))
    exacte = z0 - 0.5 * g * t**2
    euler_theorique = z0 - g * dt**2 * n * (n + 1) / 2 #courbe theorique qu'on est cense avoir 

    print(f"dt = {dt * 1e3:.0f} ms")
    print(f"  écart à la formule d'Euler : {np.max(np.abs(z - euler_theorique)):.2e}")
    print(f"  écart à t = 0,5 s (simulé)  : {(z - exacte)[-1]:+.6e}")
    print(f"  prédiction -g·t·dt/2        : {-g * t[-1] * dt / 2:+.6e}")
    plt.plot(t, z - exacte, label=f"Euler, Δt = {dt * 1e3:.0f} ms")

t, z, g = chute(2e-3, mujoco.mjtIntegrator.mjINT_RK4)
print(f"RK4 : écart max = {np.max(np.abs(z - (z0 - 0.5 * g * t**2))):.2e}")

plt.xlabel("t (s)")
plt.ylabel("z simulé − z exact (m)")
plt.legend()
plt.savefig("chute_libre_erreur.png", dpi=150)
plt.show()