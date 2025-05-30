# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Anymal C walk Coupled with Sand
#
# Shows Anymal C with a pretrained policy coupled with implicit mpm sand.
#
###########################################################################

import numpy as np
import warp as wp
from example_anymal_c_walk_coupled import Example
from omni.kit_app import KitApp

import newton


@wp.kernel
def _copy_positions(
    src: wp.array(dtype=wp.vec3),
    dest: wp.fabricarrayarray(dtype=wp.vec3),
):
    i = wp.tid()
    dest[0, i] = src[i]


@wp.kernel
def _update_body_transforms(
    src: wp.array(dtype=wp.transform),
    dest_pos: wp.fabricarray(dtype=wp.vec3d),
    dest_rot: wp.fabricarray(dtype=wp.quatf),
    ordering: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    xform = src[ordering[i]]
    dest_pos[i] = wp.vec3d(wp.transform_get_translation(xform))
    dest_rot[i] = wp.quatf(wp.transform_get_rotation(xform))


def _create_points(usdrt_stage, model, path):
    import usdrt

    # Create FSD-compatible UsdGeomPoints

    prim = usdrt_stage.DefinePrim(path, "Points")
    pts = usdrt.UsdGeom.Points(prim)

    arr = model.particle_q.numpy()
    vtarr = usdrt.Vt.Vec3fArray(arr.reshape(-1, 3))
    # widths = np.full((len(vtarr), 1), 1.0, dtype=np.float32)
    widths = model.particle_radius.numpy()[:, None] * 2.0
    vtwidths = usdrt.Vt.FloatArray(widths)
    pts.CreatePointsAttr().Set(vtarr)
    pts.CreateWidthsAttr().Set(vtwidths)

    pts_boundable = usdrt.Rt.Boundable(prim)
    pts_world_ext = pts_boundable.CreateWorldExtentAttr()
    pts_world_ext.Set(usdrt.Gf.Range3d(usdrt.Gf.Vec3d(-1, -1, -1), usdrt.Gf.Vec3d(3, 3, 3)))
    pts_boundable.CreateFabricHierarchyLocalMatrixAttr()
    pts_boundable.CreateFabricHierarchyWorldMatrixAttr()
    prim.CreateAttribute("_worldVisibility", usdrt.Sdf.ValueTypeNames.Bool, True).Set(True)
    prim.CreateAttribute("purpose", usdrt.Sdf.ValueTypeNames.Token, False).Set("default")


def _create_xform_attrs(usdrt_stage, usd_render, app):
    import usdrt

    for k, name in enumerate(usd_render.body_names):
        path = usd_render.root.GetPath().AppendChild(name)
        prim = usdrt_stage.GetPrimAtPath(str(path))

        prim.CreateAttribute("_worldPosition", usdrt.Sdf.ValueTypeNames.Double3, True).Set([k, k, k])
        prim.CreateAttribute("_worldOrientation", usdrt.Sdf.ValueTypeNames.Quatf, True).Set(usdrt.Gf.Quatf(1, 0, 0, 0))
        prim.CreateAttribute("_worldScale", usdrt.Sdf.ValueTypeNames.Float3, True).Set(usdrt.Gf.Vec3f(1, 1, 1))

    # Read back the fake transforms that we wrote above and use that to determine selection ordering
    # Why do we need to update here? no idea
    app.update()
    body_selection = usdrt_stage.SelectPrims(
        require_attrs=[
            (usdrt.Sdf.ValueTypeNames.Double3, "_worldPosition", usdrt.Usd.Access.ReadWrite),
            (usdrt.Sdf.ValueTypeNames.Quatf, "_worldOrientation", usdrt.Usd.Access.ReadWrite),
        ],
        device="cuda:0",
    )

    fpos = wp.fabricarray(data=body_selection, attrib="_worldPosition")
    fabric_ordering = wp.array(fpos.numpy()[:, 0], dtype=wp.int32, device="cuda:0")

    return fabric_ordering


def _render_bodies(usdrt_stage, state, fabric_ordering):
    import usdrt

    # Update bodies
    body_selection = usdrt_stage.SelectPrims(
        require_attrs=[
            (usdrt.Sdf.ValueTypeNames.Double3, "_worldPosition", usdrt.Usd.Access.Overwrite),
            (usdrt.Sdf.ValueTypeNames.Quatf, "_worldOrientation", usdrt.Usd.Access.Overwrite),
        ],
        device="cuda:0",
    )
    fpos = wp.fabricarray(data=body_selection, attrib="_worldPosition")
    frot = wp.fabricarray(data=body_selection, attrib="_worldOrientation")

    wp.launch(
        _update_body_transforms,
        dim=state.body_q.shape[0],
        inputs=[state.body_q, fpos, frot, fabric_ordering],
    )


def _render_particles(usdrt_stage, state):
    import usdrt

    particle_selection = usdrt_stage.SelectPrims(
        require_prim_type="Points",
        require_attrs=[
            (usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.Overwrite),
        ],
        device="cuda:0",
    )

    positions = wp.fabricarray(data=particle_selection, attrib="points", dtype=wp.vec3)
    wp.launch(
        _copy_positions,
        dim=state.particle_q.shape[0],
        inputs=[state.particle_q, positions],
    )

    # This should not be required, but no update without manually brining back to cpu :/
    particle_selection = usdrt_stage.SelectPrims(
        require_prim_type="Points",
        require_attrs=[
            (usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.ReadWrite),
        ],
        device="cpu",
    )


def _frame_prims():
    import omni.kit.commands
    import omni.kit.viewport.utility

    # Get the active viewport window
    active_vp_window = omni.kit.viewport.utility.get_active_viewport_window()
    viewport_api = active_vp_window.viewport_api

    # Get the camera path used by the viewport (e.g., "/OmniverseKit_Persp" or a user camera)
    camera_path = viewport_api.camera_path.pathString

    omni.kit.commands.execute(
        "FramePrimsCommand",
        prim_to_move=camera_path,
        prims_to_frame=["/root/ground"],
    )


def _create_stage(usd_context, options):
    import usdrt

    usd_context.new_stage()
    # if not usd_context.open_stage(args.usd_stage_path):
    #     raise RuntimeError(f"Could not open stage {args.usd_stage_path}")

    usd_stage = usd_context.get_stage()

    with wp.ScopedDevice(options.device):
        example = Example(
            urdf_path=options.urdf_path,
            voxel_size=options.voxel_size,
            particles_per_cell=options.particles_per_cell,
            tolerance=options.tolerance,
            headless=True,
        )

        usd_render = newton.utils.SimRendererUsd(example.model, path=usd_stage)
        usd_render.render_ground(size=5, plane=example.model.ground_plane_params + np.array([0.0, 0.0, 0.0, -0.0125]))

    stage_id = usd_context.get_stage_id()
    usdrt_stage = usdrt.Usd.Stage.Attach(stage_id)

    _create_points(usdrt_stage, example.model, path="/root/points")
    fabric_ordering = _create_xform_attrs(usdrt_stage, usd_render, app)
    _render_bodies(usdrt_stage, example.state_0, fabric_ordering)

    app.update()
    _frame_prims()

    return example, usdrt_stage, usd_render, fabric_ordering


def _run(app, options):
    import omni.timeline
    import omni.usd

    usd_context = omni.usd.get_context()

    timeline = omni.timeline.get_timeline_interface()
    needs_stage_reset = True

    while app.is_running():
        if timeline.is_stopped() and needs_stage_reset:
            example, usdrt_stage, usd_render, fabric_ordering = _create_stage(usd_context, options)
            needs_stage_reset = False
        elif timeline.is_playing():
            with wp.ScopedDevice(options.device):
                with wp.ScopedTimer("step", synchronize=True):
                    example.step()

                _render_bodies(usdrt_stage, example.state_0, fabric_ordering)
                _render_particles(usdrt_stage, example.state_0)

                needs_stage_reset = True

        app.update()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "urdf_path",
        type=lambda x: None if x == "None" else str(x),
        help="Path to the Anymal C URDF file from newton-assets.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=10, help="Total number of frames.")
    parser.add_argument("--voxel_size", "-dx", type=float, default=0.03)
    parser.add_argument("--particles_per_cell", "-ppc", type=float, default=3.0)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)

    options, kit_args = parser.parse_known_args()

    # KitApp is a wrapper over the omni.kit.app.IApp interface
    app = KitApp()

    # Startup kit and ask for the omni.ui extension. Kit will start it including all its dependencies
    # Add any extra command line arguments to the startup call, allowing the user to pass more to the script

    # fmt: off
    app.startup([
            "--enable", "omni.usd", 
            "--enable", "omni.usd.libs", 
            "--enable", "omni.kit.uiapp", 

            "--enable", "omni.kit.window.file",
            "--enable", "omni.kit.menu.utils",
            "--enable", "omni.kit.menu.file",
            "--enable", "omni.kit.menu.edit",
            "--enable", "omni.kit.menu.create",
            "--enable", "omni.kit.menu.common",
            "--enable", "omni.kit.context_menu",
            "--enable", "omni.kit.selection",

             "--enable", "omni.kit.widget.stage", 
             "--enable", "omni.kit.window.stage", 
             "--enable", "omni.kit.viewport.bundle", 
             "--enable", "omni.kit.viewport.rtx",
             "--enable", "omni.hydra.usdrt_delegate",
             "--enable", "usdrt.scenegraph",
            "--enable", "omni.kit.window.status_bar",
            "--enable", "omni.stats",
            "--enable", "omni.rtx.settings.core",

            "--enable", "omni.kit.window.stats",
            "--enable", "omni.kit.window.script_editor",
            "--enable", "omni.kit.window.console",
            "--enable", "omni.kit.window.preferences",
            
            "--enable", "omni.kit.window.toolbar",
            "--enable", "omni.timeline",

             "--/app/useFabricSceneDelegate=1",
             "--/renderer/multiGpu/autoEnable=false",
             ] + kit_args)

    # fmt: on

    _run(app, options)
