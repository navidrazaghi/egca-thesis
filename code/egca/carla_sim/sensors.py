"""Sensor rig (Appendix B, Table B-1): three front RGB cameras stitched to a
704 x 160 strip, one 32-beam LiDAR, IMU/speedometer and GNSS.
Requires the `carla` client package matching the simulator build (0.9.14).
"""
import math

import numpy as np

CAMERAS = [
    # name, yaw (deg). At +-60 with a 60 deg FOV the three frusta tile exactly:
    # no overlap to crop and, more importantly, no gap. This is the rig that
    # produced the published dataset the model is trained on, and the agent has
    # to present the same geometry at inference or the network sees a projection
    # it has never been shown.
    #
    # It was +-55 to match the collection rig of our own dataset, which leaves a
    # 5 deg overlap on each side. The stitcher cropped that overlap and, in
    # doing so, tore two 13 deg holes out of the panorama -- the strip ran
    # [-85,-37], [-24,+24], [+37,+85] rather than a continuous 132 deg. A model
    # trained on their seamless strip and driven with that one is being fed a
    # different scene.
    ("cam_left", -60.0), ("cam_front", 0.0), ("cam_right", 60.0),
]
CAM_W, CAM_H, CAM_FOV = 320, 160, 60
# The three cameras concatenate to 960 px spanning 180 deg; the model consumes
# the central 704, which is 132 deg. Measured seams at columns 319 and 639 of
# the published images confirm they are a plain concatenation, not a blend.
FULL_W = 3 * CAM_W
STITCH_W = 704


def sensor_blueprints(world, with_depth=False):
    """`with_depth` additionally spawns one depth camera per RGB camera, at the
    same pose and with the same intrinsics, to provide the privileged depth
    target of the auxiliary head (only needed during data collection)."""
    bp = world.get_blueprint_library()
    out = []
    for name, yaw in CAMERAS:
        cam = bp.find("sensor.camera.rgb")
        cam.set_attribute("image_size_x", str(CAM_W))
        cam.set_attribute("image_size_y", str(CAM_H))
        cam.set_attribute("fov", str(CAM_FOV))
        out.append((name, cam, dict(x=1.3, z=2.3, yaw=yaw)))
    if with_depth:
        for name, yaw in CAMERAS:
            dep = bp.find("sensor.camera.depth")
            dep.set_attribute("image_size_x", str(CAM_W))
            dep.set_attribute("image_size_y", str(CAM_H))
            dep.set_attribute("fov", str(CAM_FOV))
            out.append((name + "_depth", dep, dict(x=1.3, z=2.3, yaw=yaw)))
    lid = bp.find("sensor.lidar.ray_cast")
    lid.set_attribute("channels", "32")
    lid.set_attribute("range", "45")
    lid.set_attribute("rotation_frequency", "10")
    lid.set_attribute("points_per_second", "600000")
    out.append(("lidar", lid, dict(x=0.0, z=2.5)))
    imu = bp.find("sensor.other.imu")
    out.append(("imu", imu, dict(x=0.0, z=0.0)))
    gnss = bp.find("sensor.other.gnss")
    gnss.set_attribute("noise_lat_stddev", "0.0000045")   # ~0.5 m
    gnss.set_attribute("noise_lon_stddev", "0.0000045")
    out.append(("gnss", gnss, dict(x=0.0, z=0.0)))
    return out


def spawn_rig(world, vehicle, callback, with_depth=False):
    """Spawn all sensors attached to `vehicle`; `callback(name, data)`."""
    import carla
    actors = []
    for name, bp, pos in sensor_blueprints(world, with_depth):
        tf = carla.Transform(
            carla.Location(x=pos.get("x", 0.0), z=pos.get("z", 0.0)),
            carla.Rotation(yaw=pos.get("yaw", 0.0)))
        actor = world.spawn_actor(bp, tf, attach_to=vehicle)
        actor.listen(lambda data, n=name: callback(n, data))
        actors.append(actor)
    return actors


def stitch_cameras(images):
    """images: dict name -> H x W x 3 uint8. Concatenate the three 60 deg views
    into a 960 px, 180 deg panorama and return its central 704 px (132 deg).

    This is exactly what the training pipeline sees: the published dataset
    stores the 960 px concatenation, and the adapter centre-crops it to 704.
    Doing anything else here -- cropping overlaps, taking the first 704 of a
    768 px strip -- moves the forward axis off the image centre and changes
    which angle each column represents, which the network has no way to report
    and every reason to be broken by.
    """
    strip = np.concatenate([images["cam_left"], images["cam_front"],
                            images["cam_right"]], axis=1)
    x0 = (FULL_W - STITCH_W) // 2
    return strip[:, x0:x0 + STITCH_W]


def carla_image_to_array(image):
    a = np.frombuffer(image.raw_data, dtype=np.uint8)
    a = a.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    return a.copy()


def carla_depth_to_array(image, max_range=100.0):
    """CARLA depth image -> H x W metric depth in metres (clipped)."""
    a = np.frombuffer(image.raw_data, dtype=np.uint8)
    a = a.reshape((image.height, image.width, 4))[:, :, :3].astype(np.float32)
    # official CARLA decoding: normalized = (R + G*256 + B*256^2) / (256^3 - 1)
    norm = (a[:, :, 2] + a[:, :, 1] * 256.0 + a[:, :, 0] * 65536.0) / 16777215.0
    return np.clip(norm * 1000.0, 0.0, max_range)


def depth_to_normalized_inverse(depth_m, near=1.0, far=100.0):
    """Metric depth -> inverse depth normalized to [0, 1] (the target of the
    auxiliary head, Eq. 3.22).  Inverse depth is used because it distributes
    resolution towards the near field, which is what matters for driving."""
    d = np.clip(depth_m, near, far)
    return (1.0 / d - 1.0 / far) / (1.0 / near - 1.0 / far)


def carla_lidar_to_array(meas):
    """CARLA lidar -> N x 4 (x fwd, y left, z up, intensity) in ego frame."""
    pts = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1, 4).copy()
    pts[:, 1] *= -1.0                          # UE4 left-handed -> right-handed
    return pts


def transform_to_ego(points_world, ego_transform):
    """World -> ego frame for privileged BEV ground truth."""
    yaw = math.radians(ego_transform.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    dx = points_world[:, 0] - ego_transform.location.x
    dy = points_world[:, 1] - ego_transform.location.y
    x = c * dx + s * dy
    y = -(-s * dx + c * dy)                    # to (x fwd, y left)
    return np.stack([x, y], axis=1)
