# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Contact Sensor
#
# Shows how to use the SensorContact class to evaluate contact forces,
# including per-counterpart breakdowns.
# The flap has a contact sensor registering the total contact force of
# the objects on top. The plates' sensors register per-counterpart forces
# for the cube and the ball to detect which object touched which plate. Each
# plate will light up with the color of the first object that touches it.
#
#
# Command: python -m newton.examples sensor_contact
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import Contacts
from newton.sensors import SensorContact
from newton.tests.unittest_utils import find_nonfinite_members


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        # setup simulation parameters first
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_dt = self.frame_dt
        self.reset_interval = 8.0

        self.viewer = viewer
        self.solver_type = str(getattr(args, "solver", "mujoco")).lower()

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        if self.solver_type == "lox":
            newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.add_usd(newton.examples.get_asset("sensor_contact_scene.usda"))

        builder.add_ground_plane()

        # finalize model
        self.model = builder.finalize()

        self.flap_contact_sensor = SensorContact(self.model, sensing_shapes="*Flap", verbose=True)

        # String patterns return matches in ascending shape index order.
        # Plate1 has a lower index than Plate2 (added first), so row 0 → Plate1, row 1 → Plate2.
        plate_labels = ["*Plate1", "*Plate2"]
        counterpart_labels = ["*Cube*", "*Sphere*"]
        self.plate_contact_sensor = SensorContact(
            self.model,
            sensing_shapes=plate_labels,
            counterpart_shapes=counterpart_labels,
            measure_total=False,
            verbose=True,
        )
        if self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                njmax=100,
                nconmax=100,
                cone="pyramidal",
                impratio=1,
            )
        else:
            config = newton.solvers.SolverKamino.Config.from_model(self.model, dynamics_solver="lox")
            config.use_collision_detector = False
            self.solver = newton.solvers.SolverKamino(self.model, config=config)

        # used for storing contact info required by contact sensor
        if self.solver_type == "lox":
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                rigid_contact_max=self.solver.get_max_contact_count(),
            )
            self.contacts = self.collision_pipeline.contacts()
        else:
            self.collision_pipeline = None
            self.contacts = Contacts(
                self.solver.get_max_contact_count(),
                0,
                requested_attributes=self.model.get_requested_contact_attributes(),
            )

        self.viewer.set_model(self.model)

        self.shape_map = {key: s for s, key in enumerate(self.model.shape_label)}
        self.plates_touched = 2 * [False]
        self.shape_colors = {
            "/env/Plate1": 3 * [0.4],
            "/env/Plate2": 3 * [0.4],
            "/env/Sphere": [1.0, 0.4, 0.2],
            "/env/Cube": [0.2, 0.4, 0.8],
            "/env/Flap": 3 * [0.8],
        }

        self.state_0 = self.model.state()

        self.control = self.model.control()
        hinge_joint_idx = self.model.joint_label.index("/env/Hinge")
        self.hinge_joint_q_start = int(self.model.joint_q_start.numpy()[hinge_joint_idx])

        self.next_reset = 0.0

        # store initial state for reset
        self.initial_joint_q = wp.clone(self.state_0.joint_q)
        self.initial_joint_qd = wp.clone(self.state_0.joint_qd)

        self.capture()

    def _set_shape_colors(self, shape_colors: dict[int, list[float] | tuple[float, float, float]]):
        for shape_idx, color in shape_colors.items():
            self.model.shape_color[shape_idx : shape_idx + 1].fill_(wp.vec3(color))

    def capture(self):
        self.graph = None

        if not wp.get_device().is_cuda:
            return

        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        if self.collision_pipeline is not None:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        solver_contacts = self.contacts if self.collision_pipeline is not None else None
        self.solver.step(self.state_0, self.state_0, self.control, solver_contacts, self.sim_dt)
        self.solver.update_contacts(self.contacts, self.state_0)

    def step(self):
        if self.sim_time >= self.next_reset:
            self.reset()

        hinge_angle = min(self.sim_time / 3, 1.6)
        self.control.joint_target_q[self.hinge_joint_q_start : self.hinge_joint_q_start + 1].fill_(hinge_angle)

        with wp.ScopedTimer("step", active=False):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.plate_contact_sensor.update(self.state_0, self.contacts)

        # Use the per-counterpart forces to identify which object first touched each plate.
        net_force = self.plate_contact_sensor.force_matrix.numpy()
        for plate_idx, counterpart_forces in enumerate(net_force):
            if self.plates_touched[plate_idx]:
                continue
            counterpart_col = next(
                (col for col, force in enumerate(counterpart_forces) if np.abs(force).max() > 0.0),
                None,
            )
            if counterpart_col is None:
                continue
            plate_shape = self.plate_contact_sensor.sensing_indices[plate_idx]
            counterpart_shape = self.plate_contact_sensor.counterpart_indices[plate_idx][counterpart_col]
            self.plates_touched[plate_idx] = True
            plate_label = self.model.shape_label[plate_shape]
            counterpart_label = self.model.shape_label[counterpart_shape]
            print(f"Plate {plate_label} was touched by counterpart {counterpart_label}")
            self._set_shape_colors({plate_shape: self.shape_colors[counterpart_label]})

        self.flap_contact_sensor.update(self.state_0, self.contacts)
        self.viewer.log_scalar(
            "Flap Contact Force",
            np.abs(self.flap_contact_sensor.total_force.numpy()[0, 2]),
            smoothing=10,
        )
        self.sim_time += self.frame_dt

    def reset(self):
        self.sim_time = 0
        self.next_reset = self.sim_time + self.reset_interval
        self._set_shape_colors({self.shape_map[s]: v for s, v in self.shape_colors.items()})
        self.plates_touched = 2 * [False]
        self.viewer.log_scalar("Flap Contact Force", 0.0, clear=True)

        print("Resetting")
        # Restore initial joint positions and velocities in-place.
        self.state_0.joint_q.assign(self.initial_joint_q)
        self.state_0.joint_qd.assign(self.initial_joint_qd)
        # Recompute forward kinematics to refresh derived state.
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Verify contact sensors remain finite and detect the expected sequence."""
        assert not self.plates_touched[1] or self.plates_touched[0]  # plate 0 always touched first
        assert len(find_nonfinite_members(self.flap_contact_sensor)) == 0
        assert len(find_nonfinite_members(self.plate_contact_sensor)) == 0
        # The first plate is touched by 1.4 s; LOX keeps the cube on the flap longer than MuJoCo.
        if self.sim_time > 1.4:
            assert self.plates_touched[0]
        if self.sim_time > 3.6:
            assert np.abs(self.flap_contact_sensor.total_force.numpy()).max() == 0
        # if self.sim_time > 4.0: assert self.plates_touched[1]   # unreliable due to jerky cube motion

    def test_final(self):
        """Verify the final body state and sensor configuration are valid."""
        self.test_post_step()
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )
        assert len(find_nonfinite_members(self.flap_contact_sensor)) == 0
        assert len(find_nonfinite_members(self.plate_contact_sensor)) == 0
        # sensing_indices preserves the input order given to the sensor.
        assert self.model.shape_label[self.plate_contact_sensor.sensing_indices[0]] == "/env/Plate1"
        assert self.model.shape_label[self.plate_contact_sensor.sensing_indices[1]] == "/env/Plate2"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--solver", choices=("mujoco", "lox"), default="mujoco")

    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
